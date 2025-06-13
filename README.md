# 🔥 Fire Intensity Clustering

This project performs unsupervised clustering of fire incident data in Malaysia using MODIS satellite measurements. It leverages machine learning algorithms to uncover hidden patterns in fire intensity based on two key features: **brightness** and **fire radiative power (FRP)**.

## 📊 Objective

To analyze and group fire incidents by intensity using unsupervised learning techniques, helping identify low- and high-intensity fire zones for improved response strategies and pattern recognition.

## 🧠 Techniques Used

- **Feature Engineering**
  - Log transformation
  - Standardization
  - PCA for feature importance
- **Clustering Algorithms**
  - Fuzzy C-Means (FCM)
  - Gaussian Mixture Model (GMM)
  - Spectral Clustering
  - DBSCAN & HDBSCAN
  - Self-Organizing Maps (SOM)
  - Hierarchical Clustering

## 🧼 Data Preprocessing

- Filtered land-based fire incidents only
- Removed duplicates and invalid entries
- Handled skewed distributions with log transforms
- Applied feature scaling for optimal clustering performance

## 🧪 Evaluation Metrics

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Partition Coefficient & Partition Entropy (for FCM)

## 📈 Results Summary

- **Fuzzy C-Means (FCM)** was selected as the final model for its ability to capture gradual transitions in fire intensity.
- Spectral Clustering had the highest silhouette score but was too imbalanced.
- DBSCAN and HDBSCAN excelled in outlier detection.

## 📺 Streamlit Dashboard

Explore the interactive results here:  
👉 [Live App](https://grouping-fires-by-intensity-ubrppbquxywv26cdpypnk8.streamlit.app/)

## 📁 Dataset

- Source: [NASA FIRMS - MODIS](https://firms.modaps.eosdis.nasa.gov/download/)
- Features used: `brightness`, `frp`

## 👥 Contributors

- Chin Bao Sheng  
- Kok Ka Ket  
- Wong Jun Wei  
- Wong Rui Sean  

## 📌 Future Work

- Include spatial and temporal features
- Add real-world validation using labeled fire data
- Test scalability on larger or streaming datasets

## 📄 License

This project is for academic and research purposes. Contact the authors for reuse or extension.

---
