## Case: extract_offer_evidence

Requirements come from guests of a trip (pool, free parking, pets, ...). The
model must copy passages of the offer that speak to each requirement and leave
a requirement without quotes when the offer is silent about it.

- A quote counts as relevant when it settles or conditions the requirement,
  also with a negative or conditional answer ("no pets", "pool in summer only").
- A quote about something else (a public pool nearby when the requirement is the
  hotel's own pool) is not relevant, unless the reference uses it too.
- A requirement without reference quotes must have no candidate quotes: any quote
  there lowers the score to at most 0.5.
