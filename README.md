# UK Rental Price Indicators — data pipeline

The **primary modeling table** is now one published ONS rent estimate per
**local authority × month**, January 2015 through December 2024. It contains
316 England/Wales local authorities × 120 months = **37,920 rows**. A row is an
area-month rent statistic, **not an individual tenancy or property listing**.

The goal is to study associations between monthly private rent, home sales,
police-recorded street crime, and lagged disposable household income.

## Folder contract

| Folder | Contents |
|---|---|
| `data/raw/` | Unmodified downloaded source files. Includes ONS PIPR, ONS local-authority GDHI, ONS postcode directory, Price Paid, and street-crime CSVs. |
| `data/processed/reference/` | Normalized rent, income/population, postcode, and LSOA lookups. |
| `data/processed/sales_transactions/`, `crime_events/` | Validated event-level Parquet from the first ingestion stage. |
| `data/processed/local_authority/sales_events/`, `crime_events/` | Those events mapped to a local authority, partitioned by year. |
| `data/processed/local_authority/sales_monthly/`, `crime_force_monthly/` | Local-authority monthly aggregates. |
| `data/processed/local_authority/rent_panel/` | **Primary modeling table**, Parquet partitioned by `region_code`. |
| `data/processed/local_authority/rent_panel.csv` | Same full table in CSV format. |
| `data/processed/local_authority/by_region/` | Ten CSVs, one per English region or Wales, with local-authority rows. |
| `data/processed/regional_rent_panel/` | Earlier 10-region baseline, retained for comparison. |
| `data/failed/` | Rejected source rows and newly rejected local-authority mappings, with reasons. |
| `data/metadata/` | Source inventory, accepted/rejected counts, geographic coverage, and validation results. |

`house_hold_debt.xlsx` and `total_wealth.xlsx` remain in `raw/` but are not used.
Police outcomes, stop-and-search, British Transport Police (`btp`), and Northern
Ireland files are not in the crime baseline.

## Sources and alignment

* **Rent target:** ONS PIPR `Table 1`. Only area codes beginning `E06`, `E07`,
  `E08`, `E09`, or `W06` are retained. Each local authority has one observation
  for each of 120 months. `rent_gbp_month` is the published monthly average;
  subtype rent levels and the rent index are retained as additional measures.
* **Home sales:** HM Land Registry Price Paid records are mapped from postcode
  to local authority using the August 2024 ONS Postcode Directory (ONSPD).
  Monthly medians and counts use standard-market `category_type=A` and
  `record_status=A`; the other validated records remain in the event table.
  Median uses Spark `percentile_approx` with accuracy 10000. A sale is not a rent.
* **Crime:** police street-crime records are mapped via 2011/2021 LSOA to local
  authority. The mapping uses the LAD supported by the most ONSPD postcodes.
  If that LAD has less than 95% support, the crime event is quarantined under
  `failed/local_authority/` rather than assigned arbitrarily. Missing source
  files for a local authority's dominant territorial police force make
  `crime_count` null; `crime_count_observed` remains available for auditing.
  Even complete files can contain records without an assignable LSOA; those
  records are excluded and counted in `failed/`.
* **Income and population:** ONS *Regional gross disposable household income:
  local authorities*, Tables 1–3, provide annual total GDHI, population, and
  GDHI per person. A month in year *Y* uses the **previous year's** values
  (*Y−1*) to prevent use of same-year future information.
* **Geography revisions:** the 2026 PIPR workbook uses `E08000038` (Barnsley)
  and `E08000039` (Sheffield), while the 2024 ONSPD/GDHI workbooks use
  `E08000016` and `E08000019` respectively. These two code mappings are explicit
  and name-checked in `prepare_local_authority.py`. Historic postcodes and
  boundary changes can still cause some unmatched events; the affected counts
  are reported.

All monetary values are nominal GBP, not inflation-adjusted. Region means the
nine English regions plus Wales; a local authority is the finer modeling unit.

## Modeling table and split

Primary key: `lad_code, month` (`YYYY-MM`). Other identifiers are `lad_name`,
`region_code`, `region_name`. The target is `rent_gbp_month`. Features include
sale medians/counts, crime counts/rate and coverage flag, prior-year GDHI and
population, plus optional rent subtype measures. Do **not** use same-month rent
subtypes or `rent_index` as predictors for the overall rent target: they are
parallel measures of the target and would leak its value.

| Period | Split | Expected rows |
|---|---|---:|
| 2015 | `history` | 3,792 |
| 2016–2022 | `train` | 26,544 |
| 2023 | `validation` | 3,792 |
| 2024 | `test` | 3,792 |

There is no global imputation, scaling, or feature selection in ingestion.
Model-side transformations must be fitted on training rows only. Sales and
crime are contemporaneous monthly variables, suitable for an association
study. For genuine forward prediction, lag those features and account for
their publication delay. Rows within an authority and neighboring months are
correlated; 37,920 rows are not 37,920 independent rental contracts.

## Reproduce

Requires Python ≥3.10, Java 17, PySpark 3.5, and openpyxl. From project root:

```bash
python3 -m pip install -r requirements.txt
python3 source_inventory.py
python3 prepare_reference.py
python3 run_pipeline.py --resume
python3 prepare_local_authority.py
python3 build_local_authority_panel.py --resume
python3 validate_local_authority.py
```

The first two ingestion scripts create validated source events and retain the
earlier region baseline. The local-authority scripts remap those events, so a
rerun need not re-read all raw CSVs. Use `--only-year YYYY` on the relevant Spark
script to regenerate an interrupted year before `--resume`. On machines with
several Python installations, set `PYSPARK_PYTHON` and
`PYSPARK_DRIVER_PYTHON` to the same interpreter.

`data/metadata/local_authority_reference.json`,
`local_authority_year_YYYY.json`, `local_authority_crime_coverage.csv`,
`local_authority_panel_quality.json`, and
`local_authority_independent_validation.json` provide inspectable evidence.
Validation checks the 37,920-row grain, key uniqueness, time split, annual lag,
income coverage, region CSV counts, and missing-crime handling.

Sources: [ONS PIPR](https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics), [ONS local-authority GDHI](https://www.ons.gov.uk/economy/regionalaccounts/grossdisposablehouseholdincome/datasets/regionalgrossdisposablehouseholdincomelocalauthorities), [HM Land Registry Price Paid](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads), [Police.uk data](https://data.police.uk/data/), [ONS Postcode Directory August 2024](https://www.data.gov.uk/dataset/f888a19a-ceca-4502-8090-bf6e5aa71d1e/ons-postcode-directory-august-2024-for-the-uk).

