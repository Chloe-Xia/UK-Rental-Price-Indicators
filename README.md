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
| `data/raw/` | Unmodified downloaded source files. Includes ONS PIPR, ONS local-authority GDHI, ONS postcode directory, Price Paid, street-crime CSVs, and Home Office CSP recorded crime (`Home Office Crime/`). |
| `data/processed/reference/` | Normalized rent, income/population, postcode, and LSOA lookups. |
| `data/processed/sales_transactions/`, `crime_events/` | Validated event-level Parquet from the first ingestion stage. |
| `data/processed/local_authority/sales_events/`, `crime_events/` | Those events mapped to a local authority, partitioned by year. |
| `data/processed/local_authority/sales_monthly/`, `crime_force_monthly/` | Local-authority monthly aggregates. |
| `data/processed/local_authority/rent_panel/` | **Primary modeling table**, Parquet partitioned by `region_code`. |
| `data/processed/local_authority/rent_panel.csv` | Same full table in CSV format. |
| `data/processed/local_authority/by_region/` | Ten CSVs, one per English region or Wales, with local-authority rows. |
| `data/processed/local_authority/features/`, `features.csv` | **Feature table** for ML and analysis: same 37,920 rows plus engineered features (see below). |
| `data/processed/reference/lad_geography.csv`, `lad_neighbours.csv` | Static local-authority attributes from ONSPD and the 5 nearest local authorities. |
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
  records are excluded and counted in `failed/`. A published file is also
  treated as missing (`crime_partial_upload`) when the force-month has under
  50% of the force's median monthly records (Greater Manchester 2022-08,
  Dyfed-Powys 2023-05 and 2024-01..04) or the authority has under 25% of its
  own median (Leicestershire 2020-03, where the source left most records
  without an LSOA). Greater Manchester published no street crime from 2019-07
  to 2024-12.
* **Recorded crime:** Home Office *Police recorded crime Community Safety
  Partnership open data* (quarterly, by offence; fraud excluded) is complete for
  every force, including Greater Manchester. `prepare_home_office_crime.py` maps
  CSPs to rent local authorities: predecessor CSPs merge into successor
  authorities, and CSPs covering several authorities (e.g. South
  Nottinghamshire, Western Suffolk, Cwm Taf) are split by population and
  flagged `recorded_crime_allocated`. Offences with no CSP ("Unassigned", up to
  a third of Kent in 2024/25 and nearly all of Suffolk in 2015-Q2/Q3) are
  redistributed within force, quarter, and offence group using each CSP's share
  over the previous four quarters. It excludes anti-social behaviour, which
  police.uk includes.
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

## Feature table

`build_features.py` adds features to the panel without changing its grain.
`data/metadata/feature_dictionary.csv` documents every column with a `role`
and two flags: `use_association` (contemporaneous study of what rent co-moves
with) and `use_forecast` (only information available before month *t*).

| Group | Examples |
|---|---|
| Target variants | `log_rent`, `rent_yoy_log_change`, `rent_relative_to_national`, `rent_relative_to_region` — never predictors |
| Time | `t_index`, `month_sin/cos`, `covid_restrictions`, `transaction_tax_holiday` (England/Wales dates differ) |
| Geography (static) | centroid, `dist_to_london_km`, `urban_postcode_share`, `imd_deprivation_score`, `lad_type`, `is_london` |
| Income/population (Y−1) | `log_gdhi_per_head`, `log_population`, `population_growth`, `gdhi_per_head_growth` |
| Housing market | trailing 3/12-month sale medians, sales per 1000 residents, property-type / new-build / leasehold shares, `price_to_income_ratio` |
| Street crime (police.uk) | `crime_per_1000_roll12`, eight crime-group rates incl. anti-social behaviour, `crime_yoy_log_change` |
| Recorded crime (Home Office) | `recorded_crime_per_1000_roll4q`, nine offence-group rates, `recorded_crime_yoy_log_change`; `*_prevq` versions use only completed quarters |
| Spatial | mean of the 5 nearest local authorities: lagged rent, sale prices, crime |
| Lagged (forecast) | rent lags 1/3/12, lagged rent growth, `*_lag1` copies of market and crime features |

Rolling windows are trailing and require full windows (crime: at least 6 valid
months), so 2015 (`history`) is mostly null for those features. Street-crime
gaps of at most 12 consecutive months (176 rows: Devon and Cornwall, Dyfed-Powys,
Staffordshire, and four single months) are imputed as the same month a year
earlier times the region's year-on-year change and flagged `crime_imputed`;
on observed months this rule has a median absolute error of 6.9%. Greater
Manchester's 66-month gap (660 rows) is not imputed: use the recorded-crime
features, which are complete for every row and correlate 0.94 with street
crime elsewhere. Any other imputation, scaling, and categorical encoding are
left to the model and must be fitted on `train` rows.
`data/metadata/feature_quality.json` reports null counts by split and checks.

## Reproduce

Requires Python 3.10–3.11, Java 17, PySpark 3.5, and openpyxl. On Windows,
Spark also needs `HADOOP_HOME` pointing to a folder whose `bin/` contains
`winutils.exe` and `hadoop.dll` (Hadoop 3.3.x). From project root:

```bash
uv sync
uv run python source_inventory.py
uv run python prepare_reference.py
uv run python run_pipeline.py --resume
uv run python prepare_local_authority.py
uv run python prepare_home_office_crime.py
uv run python build_local_authority_panel.py --resume
uv run python validate_local_authority.py
uv run python prepare_lad_geography.py
uv run python build_features.py
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
income coverage, region CSV counts, missing and partial street-crime handling,
and recorded-crime completeness. `recorded_crime_reference.json` reports the
CSP mapping, redistributed unassigned offences, and allocated authorities.

Sources: [ONS PIPR](https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics), [ONS local-authority GDHI](https://www.ons.gov.uk/economy/regionalaccounts/grossdisposablehouseholdincome/datasets/regionalgrossdisposablehouseholdincomelocalauthorities), [HM Land Registry Price Paid](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads), [Police.uk data](https://data.police.uk/data/), [Home Office police recorded crime CSP open data](https://www.gov.uk/government/statistical-data-sets/police-recorded-crime-and-outcomes-open-data-tables), [ONS Postcode Directory August 2024](https://www.data.gov.uk/dataset/f888a19a-ceca-4502-8090-bf6e5aa71d1e/ons-postcode-directory-august-2024-for-the-uk).

