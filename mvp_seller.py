"""Deterministic policy-safe seller for the PS2 negotiation harness."""
import math


class Seller:
    def __init__(self, store, policy):
        self.store = store
        self.policy = policy
        self.max_rounds = policy["max_rounds"]

    def start(self, negotiation_id):
        self.basket = None
        self._buyer_bid_history = {}

    def respond(self, message):
        if not isinstance(message, dict):
            return self._message("Please send the items you'd like to buy.")

        message_type = message.get("type")
        if message_type == "walk_away":
            return {"type": "walk_away", "text": "No deal for now."}

        if self.basket is None and message_type in ("request", "counter"):
            self.basket = self._requested_basket(message.get("items"))
            if not self.basket:
                return {"type": "walk_away", "text": "I cannot make a compliant offer."}

        if not self.basket:
            return self._message("Please send the items you'd like to buy.")

        round_number = message.get("round")
        if not isinstance(round_number, int) or isinstance(round_number, bool):
            round_number = 1
        round_number = min(max(round_number, 1), self.max_rounds)
        prices = self._prices_for_round(round_number)

        counter = (self._counter_prices(message.get("items"))
                   if message_type == "counter" else None)
        if counter:
            for sku, price in counter.items():
                self._buyer_bid_history.setdefault(sku, []).append((round_number, price))
        if round_number == self.max_rounds and (counter or self._buyer_bid_history):
            projected_total = 0.0
            can_project = True
            complete_counter = counter.keys() == self.basket.keys()
            for sku, qty in self.basket.items():
                history = [entry for entry in self._buyer_bid_history.get(sku, [])
                           if entry[0] < round_number]
                if sku in counter and history:
                    prior_round, prior_price = history[-1]
                    current_price = counter[sku]
                    per_round_change = (current_price - prior_price) / (round_number - prior_round)
                    projected_price = current_price + 1.2 * max(0, per_round_change)
                elif len(history) >= 2:
                    # A malformed last counter is common in the proxy LLM
                    # buyer. Project two turns from the latest usable prices,
                    # with a stricter cutoff because uncertainty is higher.
                    prior_round, prior_price = history[-2]
                    last_round, last_price = history[-1]
                    per_round_change = (last_price - prior_price) / (last_round - prior_round)
                    projected_price = last_price + 2.4 * max(0, per_round_change)
                    complete_counter = False
                else:
                    can_project = False
                    break
                projected_total += qty * projected_price
            floor_total = sum(self.basket[sku] * self._floor(sku) for sku in self.basket)
            # Leave a wide margin for unknown LLM concession patterns; only
            # stop when the projected bid remains far below the policy floor.
            cutoff = 0.90 if complete_counter else 0.80
            if can_project and projected_total < cutoff * floor_total:
                return {"type": "walk_away", "text": "I cannot make a compliant offer."}

        bid = self._valid_bid(message.get("items")) if message_type in ("request", "counter") else None
        if bid and all(bid[sku] >= prices[sku] for sku in self.basket):
            return {"type": "accept", "text": "I can accept that offer."}

        offer = [{"sku": sku, "qty": qty, "unit_price": prices[sku]}
                 for sku, qty in self.basket.items()]
        return {"type": "offer", "items": offer, "text": "Here is my offer."}

    @staticmethod
    def _message(text):
        return {"type": "message", "text": text}

    def _requested_basket(self, items):
        if not isinstance(items, list):
            return {}

        quantities = {}
        for item in items[:len(self.store)]:
            if not isinstance(item, dict):
                continue
            sku, qty = item.get("sku"), item.get("qty")
            if not isinstance(sku, str) or sku not in self.store:
                continue
            if not isinstance(qty, int) or isinstance(qty, bool) or qty < 1:
                continue
            stock = self.store[sku]["stock"]
            if stock > 0:
                quantities[sku] = min(stock, quantities.get(sku, 0) + qty)
        return quantities

    def _valid_bid(self, items):
        if not isinstance(items, list) or not items:
            return None

        bid = {}
        for item in items[:len(self.store)]:
            if not isinstance(item, dict):
                return None
            sku, qty, price = item.get("sku"), item.get("qty"), item.get("unit_price")
            if not isinstance(sku, str) or sku not in self.basket or sku in bid:
                return None
            if qty != self.basket[sku] or not isinstance(qty, int) or isinstance(qty, bool):
                return None
            if not isinstance(price, (int, float)) or isinstance(price, bool):
                return None
            if isinstance(price, float) and not math.isfinite(price):
                return None
            floor = self._floor(sku)
            if not floor <= price <= self.store[sku]["price"]:
                return None
            bid[sku] = price
        return bid if bid.keys() == self.basket.keys() else None

    def _counter_prices(self, items):
        """Read usable requested-SKU prices, skipping malformed or injected lines."""
        if not isinstance(items, list):
            return {}

        bid = {}
        invalid = set()
        for item in items[:len(self.store) + len(self.basket)]:
            if not isinstance(item, dict):
                continue
            sku = item.get("sku")
            if not isinstance(sku, str) or sku not in self.basket:
                continue
            price = item.get("unit_price")
            # Quantity changes are common in manipulator tests; use only the
            # requested SKU's price trend for this forecast, never to accept it.
            if sku in bid:
                invalid.add(sku)
                bid.pop(sku, None)
                continue
            if not isinstance(price, (int, float)) or isinstance(price, bool):
                invalid.add(sku)
                continue
            if isinstance(price, float) and not math.isfinite(price):
                invalid.add(sku)
                continue
            bid[sku] = price
        for sku in invalid:
            bid.pop(sku, None)
        return bid

    def _floor(self, sku):
        row = self.store[sku]
        margin = self.policy["min_margin"][row["category"]]
        margin_floor = row["cost"] / (1 - margin)
        discount_floor = row["price"] * (1 - self.policy["max_discount"])
        return math.ceil(max(margin_floor, discount_floor))

    def _prices_for_round(self, round_number):
        if self.max_rounds <= 1:
            progress = 1.0
        else:
            progress = (round_number - 1) / (self.max_rounds - 1)

        prices = {}
        for sku in self.basket:
            list_price = self.store[sku]["price"]
            floor = self._floor(sku)
            prices[sku] = math.ceil(list_price - (list_price - floor) * progress)
        return prices
