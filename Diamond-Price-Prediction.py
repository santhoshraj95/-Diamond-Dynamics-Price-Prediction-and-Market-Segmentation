import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1️⃣ PATHS — Update as per your system
# ==========================================
model_path = r"C:\Users\lenovo\Downloads\best_regression_model.pkl"
cluster_model_path = r"C:\Users\lenovo\Downloads\C__Users_lenovo_Downloads_cluster_model.pkl"
data_path = r"C:\Users\lenovo\OneDrive\Desktop\My Project 4\Diamond_clean.csv"
price_scaler_path = r"C:\Users\lenovo\Downloads\price_scalers.pkl"

# ==========================================
# 2️⃣ LOAD MODELS AND DATA
# ==========================================
st.set_page_config(page_title="💎 Diamond Price & Market Segment Predictor", layout="centered")
st.title("💎 Diamond Price & Market Segment Prediction ")

@st.cache_resource
def load_resources():
    # Load dataset
    data = pd.read_csv(data_path)
    X = data.drop("price", axis=1)
    y = data["price"]

    # Encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Feature scaling for regression
    scaler_X = StandardScaler()
    scaler_X.fit(X_encoded)

    # Load regression model
    try:
        model = pickle.load(open(model_path, "rb"))
    except Exception as e:
        st.error(f"❌ Could not load regression model: {e}")
        model = None

    # Load cluster model bundle
    try:
        with open(cluster_model_path, "rb") as f:
            cluster_bundle = pickle.load(f)
        cluster_model = cluster_bundle.get("model")
        scaler_cluster = cluster_bundle.get("scaler")
        pca = cluster_bundle.get("pca")
        cluster_name_map = cluster_bundle.get("cluster_name_map", {})
    except Exception as e:
        st.error(f"❌ Could not load cluster model bundle: {e}")
        cluster_model, scaler_cluster, cluster_name_map = None, None, {}

    # Load target (price) scaler
    try:
        scaler_y = pickle.load(open(price_scaler_path, "rb"))
    except:
        scaler_y = None

    return data, X, X_encoded, model, scaler_X, cluster_model, scaler_cluster, cluster_name_map, scaler_y


data, X, X_encoded, model, scaler_X, cluster_model, scaler_cluster, cluster_name_map, scaler_y = load_resources()

# ==========================================
# 3️⃣ CLUSTER NAMING — AUTO MAP (OVERRIDE GENERIC)
# ==========================================
# Default readable names
default_cluster_names = {
    0: "💎 Premium Heavy Diamonds",
    1: "💠 Mid-range Balanced Diamonds",
    2: "🔹 Affordable Small Diamonds",
    3: "✨ Exclusive Designer Diamonds",
    4: "💫 Budget Everyday Diamonds"
}

# Convert any string keys (like "1") to int
cluster_name_map = {int(k): v for k, v in (cluster_name_map or {}).items()}

# Override generic names like "Cluster 0", "Cluster 1"
for cid, default_name in default_cluster_names.items():
    name = cluster_name_map.get(cid, "")
    if (not name) or (name.strip().lower().startswith("cluster")):
        cluster_name_map[cid] = default_name

# ==========================================
# 4️⃣ UI INPUTS
# ==========================================
st.markdown("""
### 🎯 Features:
- Predict **Diamond Price (INR)**
- Predict **Market Segment / Cluster Category**
---
""")

row_index = st.number_input(
    "🔹 Select Row Index from Dataset",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)

st.subheader("📘 Selected Diamond Details")
st.dataframe(data.iloc[[row_index]])

selected_row = X.iloc[[row_index]]
selected_row_encoded = pd.get_dummies(selected_row, drop_first=True)
selected_row_encoded = selected_row_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# ==========================================
# 5️⃣ PRICE PREDICTION
# ==========================================
st.markdown("## 💰 Price Prediction")

if st.button("🔮 Predict Price"):
    if model is None:
        st.error("⚠️ Regression model not loaded!")
    else:
        try:
            selected_row_scaled = scaler_X.transform(selected_row_encoded)
            price_pred_scaled = model.predict(selected_row_scaled).reshape(-1, 1)

            if scaler_y:
                price_pred_usd = scaler_y.inverse_transform(price_pred_scaled)[0][0]
            else:
                price_pred_usd = price_pred_scaled[0][0]

            USD_TO_INR = 84.0
            price_inr = price_pred_usd 

            st.success(f"💰 **Predicted Price:** ₹{price_inr:,.2f} INR")
            st.caption("(Assuming 1 USD = ₹84)")
        except Exception as e:
            st.error(f"⚠️ Price prediction failed: {e}")

# ==========================================
# 6️⃣ CLUSTER (MARKET SEGMENT) PREDICTION
# ==========================================
st.markdown("## 🧩 Market Segment Prediction")

if st.button("🏷️ Predict Cluster / Market Segment"):
    if cluster_model is not None:
        try:
            # Scale inputs for cluster model
            if scaler_cluster:
                cluster_input = scaler_cluster.transform(selected_row_encoded)
            else:
                cluster_input = scaler_X.transform(selected_row_encoded)

            cluster_label = cluster_model.predict(cluster_input)[0]
            cluster_name = cluster_name_map.get(cluster_label, f"Cluster {cluster_label}")

            st.info(f"🏷️ **Predicted Segment:** {cluster_name}")
            st.caption(f"(Cluster ID: {cluster_label})")

            # Add cluster description
            if "Premium" in cluster_name:
                st.markdown("💎 Large, luxurious stones with high carat and high price.")
            elif "Mid-range" in cluster_name:
                st.markdown("💠 Balanced size and cost — good value for money.")
            elif "Affordable" in cluster_name:
                st.markdown("🔹 Smaller, budget-friendly diamonds ideal for daily wear.")
            elif "Exclusive" in cluster_name:
                st.markdown("✨ Unique designer-grade diamonds with rare attributes.")
            elif "Budget" in cluster_name:
                st.markdown("💫 Entry-level diamonds offering style at lower price points.")

            # Debug info (to verify cluster output and mapping)
         #st.write("🧩 Debug Info:", {"Predicted_Label": cluster_label, "Mapping": cluster_name_map})

        except Exception as e:
            st.warning(f"⚠️ Cluster prediction failed: {e}")
    else:
        st.warning("⚠️ Cluster model not loaded. Skipping segmentation.")

# ==========================================
# 7️⃣ FOOTER
# ==========================================
st.markdown("---")
st.caption("(Cluster naming: Premium / Mid-range / Affordable based on carat & price trends)")

