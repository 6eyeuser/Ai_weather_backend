## 🧠 AI Training & Data Pipeline (R&D)

While the live dashboard provides a seamless administrative UX, the core of this project relies on a custom PyTorch architecture trained on raw multidimensional meteorological data. 

**[🔗 View the Full Data Engineering & Training Pipeline on Kaggle](https://www.kaggle.com/code/kartikkumargggg/notebook04fa9428da)**

### The Scientific Approach
Standard computer vision models are built for flat 3-channel RGB images. However, real-world meteorological anomaly detection requires multidimensional analysis. To solve the problem statement, we bypassed standard image classification and engineered a custom pipeline utilizing **Copernicus ERA5 Reanalysis Data**.

### Pipeline Highlights
* **Multidimensional Data Extraction:** Interfaced directly with the Copernicus Climate Data Store (CDS) API to extract continuous NetCDF tensors (`.nc`), isolating a specific temporal bounding box over the Indian subcontinent.
* **Custom PyTorch Architecture:** Fundamentally altered a MobileNet architecture, rewriting the initial convolutional layers to accept **4-channel atmospheric grid data** instead of standard visual inputs.
* **Feature Engineering:** The model processes four critical atmospheric variables simultaneously to detect severe convective thunderstorms:
  1. `10m_u_component_of_wind` (Zonal Wind)
  2. `10m_v_component_of_wind` (Meridional Wind)
  3. `mean_sea_level_pressure` (MSLP)
  4. `convective_available_potential_energy` (CAPE)
* **Threshold Classification:** Engineered a custom pseudo-labeling system to trigger "Severe" threat classifications based on anomalous CAPE thresholds (> 1000 J/kg), cross-referenced with localized pressure drops and wind shear matrices. The final `.pt` weights are deployed directly to our FastAPI Render backend for live inference.