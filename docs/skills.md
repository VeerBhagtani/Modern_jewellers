# Questions, GST and leads

Arouse is a 5.5M-parameter model. It cannot memorise facts or do reliable arithmetic, so these skills follow the same rule as reminders: **the model understands, code computes.**
- The model decides what the user wants and copies the relevant words.
- Deterministic tools do the lookup, the maths and the filtering.
- The runtime checks the answer against the tool result.

## Easy questions: `kb.search`

`arouse/agent/knowledge.json` holds 53 short, hand-written answers covering:
- **GST:** rates under GST 2.0 from 22 September 2025, gold and making charges, dairy, registration limits, composition, GSTIN, input tax credit, GSTR-1/3B, QRMP, late fees, e-way bills, e-invoices, HSN
- **Gold and jewellery:** karat and purity, 916, BIS hallmark and HUID, making charges, price formula, tola, sterling silver
- **Dairy:** fat and SNF, cow vs buffalo milk, milk storage
- **Small business:** leads, margin, markup, profit, break-even, turnover, cash flow, working capital, inventory, invoices, PAN, Udyam, UPI, lakh and crore

Each answer names its source, and the GST facts were checked against current sources (listed in the file).

How a question is answered:
1. The model sends the user's question to `kb.search` (the question text is copied, never written by the model).
2. A deterministic search (`arouse/agent/knowledge.py`: stemming, IDF-weighted coverage, best-phrasing precision) returns one entry or "not found".
3. Arouse answers with that text. The runtime checks the reply against the tool result and corrects it if it differs.
4. Anything the knowledge base doesn't cover gets: "I don't know that yet. I can answer questions about GST, gold and jewellery, dairy and running a small business."

**Search quality** (`tests/test_business.py`):
- 100% of the knowledge base's own phrasings find their entry.
- 27 of 33 held-out questions (written after tuning, never tuned on) find the right entry. Five get "I don't know that yet"; one gets the wrong entry.
- 0 of 306 general-knowledge questions (capital cities, sports, science, …) are wrongly answered.

Live prices (gold rate, news) are deliberately not answered: Arouse works offline.

## GST calculation: `gst.calculate`

```
"Calculate GST on 2.5 lakh at 3%"  ->  gst.calculate {"amount": "2.5 lakh", "rate": "3"}
                                    ->  GST at 3% on ₹2,50,000 is ₹7,500 (CGST ₹3,750 + SGST ₹3,750). Total: ₹2,57,500.
```

- **Amount and rate:** both are copied exactly as the user wrote them ("₹5,000", "Rs. 12,500", "50k", "1.2 crore", "18%"). The runtime rejects values the user never wrote.
- **Arithmetic:** done in integer paise, rounding half up. CGST and SGST are half each.
- **Inclusive prices:** "1180 includes 18% GST" gives the GST inside the price and the price before GST.
- **Missing rate:** Arouse asks "Which GST rate should I use? For example 3% for gold, 5% or 18%."

## Finding leads: `leads.find`

Leads come from your own customer list, `customers.csv`, with the columns `name, city, interest, last_purchase, total_spent, birthday, anniversary`. Arouse has no internet access, so it does not search the web.

"Find me leads" first gets three suggestions, plus an invitation to give your own:
> I can find leads in 3 ways: 1) past customers who haven't bought in 3 months, 2) customers with a birthday or anniversary in the next 30 days, 3) your top customers, to ask for referrals. Which one? Or tell me your own idea.

| Your reply | What Arouse does |
|---|---|
| "1", "the second one", "referrals", … | runs that method: `inactive` (90+ days since the last purchase), `occasions` (birthday or anniversary within 30 days) or `top` (highest total spend) |
| your own idea, such as "customers in Pune who like gold" | **method 4, your method:** `custom`. Your words are copied into the query, and every customer matching all its keywords (name, city, interest) is returned |
| "none of these" | asks "Okay. Tell me your idea: how should I look for leads?", then uses your method |

If you name a method directly ("Who are my best customers?", "Find leads in Pune"), Arouse skips the question.

**Results:**
- Results are sorted (by spend, or by how soon the occasion is); the first three are shown, with the total count.
- With no `customers.csv`, Arouse says what file it needs and with which columns.
- The website and the local app ship a sample `customers.csv` whose dates are relative to today.

## Context length

Lead lists and GST answers made episodes longer than the original 768-token context. The v4 model is fine-tuned at 1,024 tokens:
- Positions use RoPE, so the weights are unchanged.
- `init_from` allows a different context length.
