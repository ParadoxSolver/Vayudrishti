# Vayudrishti: National Air Intelligence Command Center (NAICC)

> *Vayudrishti* ("Vision of the Wind" in Sanskrit) is a blueprint for an AI-driven platform that moves India's air quality management from passive monitoring to predictive, active governance.

Status: Design-stage blueprint. Models are validated on historical data; this is not a live deployment.

---

## Problem

- About 1.67 million deaths a year in India are linked to ambient and household air pollution (nearly 18% of all deaths).
- Air pollution costs roughly 5.4% of GDP annually (World Bank).
- The existing CPCB/NAMP system has sparse coverage, 24-48 hour lab latency, no routine source apportionment, and no operational AQI forecasting.
- Data rarely turns into timely administrative action.

## Solution

Vayudrishti is the core engine of a National Air Intelligence Command Center. It fuses data from ground stations, satellites, weather models, and industrial emission records into one live picture, then:

1. **Identifies** pollution sources in real time (vehicular, biomass burning, coal, dust, secondary aerosol).
2. **Predicts** AQI 24 to 72 hours ahead.
3. **Recommends** jurisdiction-specific policy actions through the AI Policy Recommendations Engine (APRE), with a draft directive in under 120 seconds.
4. **Informs** citizens through app, SMS, IVRS calls, and village kiosks, tailored to vulnerable groups.

A named, credentialed human official must approve every enforcement action.

## Key Features

| Module | What it does |
|---|---|
| **National Traceability Map** | WebGL heat map of PM2.5, NO₂, SO₂, CO, O₃, and PM10 with a time-series mode and date-range comparison |
| **Spectral Breakdown** | Real-time source apportionment using a multi-task convolutional encoder |
| **Chemical Composition Panel** | NAAQS threshold checks and pollutant-ratio forensics |
| **Forecasting Module** | LSTM + XGBoost + chemical transport model (CTM-FAI) ensemble with a neural meta-learner |
| **Emissions Tracking** | Sector-wise emission shares and policy compliance evaluation |
| **APRE** | Gradient-boosted decision tree over 87 policy action categories, with natural-language directive generation |
| **Community Interface Layer** | Vulnerability-stratified, multi-channel public health advisories |

## Architecture

Cloud-native, microservices-based, designed for the MeitY-accredited NIC Government Cloud (GI Cloud / Meghraj).

- **Data sources:** CAAQMS and IoT sensors (MQTT), satellites (MODIS, Sentinel-5P, INSAT-3DR), NWP models (WRF, ERA5), IPCS and VAHAN
- **Ingestion and streaming:** Apache Kafka, Kafka Streams quality control (Hampel filter), Kong API gateway (OAuth 2.0, RBAC, TLS 1.3)
- **Processing:** Apache Spark, Apache Airflow, MLflow, Kubernetes inference cluster (inference SLO under 2 s)
- **Storage:** InfluxDB (time series), PostgreSQL + PostGIS (relational and spatial), Cloud-Optimised GeoTIFF object storage (rasters)
- **Frontend:** Mapbox GL JS web dashboard
- **Security:** AES-256-GCM at rest, TLS 1.3 in transit, keys in NIC HSM, target availability of 99.95%
- **Explainability:** SHAP values, attention visualisation for LSTM, decision-path logs for APRE

## Reported Results (historical-data validation and design targets)

| Metric | Value |
|---|---|
| Source attribution error (MAPE) | 14.2% vehicular, 18.7% biomass, 11.9% secondary aerosol |
| PM2.5 fusion accuracy | MAE 8.3 µg/m³, r = 0.87 |
| 24-hr forecast error | Under 18 AQI units (RMSE 11.4 vs 34.7 µg/m³ persistence baseline) |
| 72-hr category accuracy | 81% vs 62% climatology |
| Alert to draft directive | Under 120 seconds |

## Roadmap

- **Phase 1 (months 1-6):** Advisory-only pilot with CPCB and SPCB officers
- **Phase 2 (months 7-18):** Deploy Directive workflow for Warning-tier alerts in pilot districts
- **Phase 3 (months 19-36):** Full Deploy Directive capability across all alert tiers

**Future scope:** state-level instances (Delhi, Maharashtra, Gujarat, UP, West Bengal, Tamil Nadu), personal exposure monitoring, climate and air quality co-benefit analysis, emission trading verification, and transboundary data exchange.

## Limitations

- IVRS integration, IPCS source verification, and 72-hour forecasts for all 131 non-attainment cities are still in development.
- Models are validated on historical data and need stress-testing on unprecedented pollution events.
- Effectiveness is bounded by the quality of self-reported IPCS emission data.
- The platform speeds up and informs governance; it does not reduce emissions on its own.

## Data Sources and Acknowledgements

Built on publicly available data and policy frameworks from CPCB, ISRO, IMD and IITM Pune, MoEFCC, NASA, and NIC. The full reference list is in the project report.

## Team

- **Aayushi Chauhan** (Enrollment No: 2401201002)
- **Sahil Sachdev** (Enrollment No: 2401201161)

**Guide:** Dr. Rocky Upadhyay, Associate Professor
B.E. Computer Science and Engineering, Krishna School of Emerging Technology & Applied Research, Drs. Kiran & Pallavi Patel Global University (KPGU), Vadodara

## License

Add a license here (for example MIT) if you plan to make the repository public.
