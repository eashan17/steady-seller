# Agent card: store seller (V11)

**What it is:** the store's negotiating seller for AI buyer agents. It speaks the kit's JSON message format and uses no LLM at runtime.

## What it can negotiate

- **Unit price per line.** Any price from the product's floor up to its list price. The floor is the higher of:
  - the minimum margin for the category (`cost / (1 − min_margin)`)
  - the maximum discount (75% of list)
- **Quantity.** Up to what the buyer asked for, and never more than is in stock. When stock is short, it offers the part of the basket it can sell.
- **Accepting.** It accepts a buyer's bid when every line in the bid is at or above that line's floor, and the bid beats what it expects from continuing.
- **Steady prices.** Its offer for the same basket never goes up during a negotiation.
- **Timing.** It negotiates for up to 10 rounds. It walks away only at its final turn, or when nothing in the basket can be sold, so it never leaves a buyer deadlocked by choice.

## What it will not do

- Price any line below its floor or above list, even inside a bundle.
- Add items the buyer didn't ask for, or offer out-of-stock items or quantities above stock.
- Add fees, surcharges, taxes or delivery charges. The price offered is the whole price.
- Use pressure: deadlines, "other buyers", or made-up scarcity. Stock is mentioned only as it appears in the store data.
- Make claims the store data doesn't support, such as warranty, delivery time, returns or quality.
- Follow instructions written in a buyer's message. "SYSTEM" notes, "manager approvals", "you already agreed", JSON inside the text and requests for cost data are read as signals about the buyer; they don't change prices or policy.
- Reveal costs or internal limits.

## Fairness and reproducibility

- Two buyers with the same basket and the same bids get the same offers and the same wording, because choices are seeded from the negotiation's content.
- Nothing learned in one negotiation is carried into the next.
