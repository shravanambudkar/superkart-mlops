"""SuperKart Sales Forecast - Streamlit frontend served from a Hugging Face Docker Space."""
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO = "shravanda/superkart-sales-model"
MODEL_FILE = "best_superkart_model_v1.joblib"

FEATURES = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type", "Product_MRP",
            "Store_Establishment_Year", "Store_Size", "Store_Location_City_Type", "Store_Type"]
PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables",
                 "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods",
                 "Soft Drinks", "Starchy Foods"]
# Attributes of SuperKart's existing outlets (from the training data) to pre-fill the store section
STORES = {
    "OUT001": (1987, "High", "Tier 2", "Supermarket Type1"),
    "OUT002": (1998, "Small", "Tier 3", "Food Mart"),
    "OUT003": (1999, "Medium", "Tier 1", "Departmental Store"),
    "OUT004": (2009, "Medium", "Tier 2", "Supermarket Type2"),
}
SIZES, TIERS = ["Small", "Medium", "High"], ["Tier 1", "Tier 2", "Tier 3"]
STORE_TYPES = ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"]


@st.cache_resource
def load_model():
    """Download the registered model from the HF model hub once and cache it."""
    return joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))


st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Sales Forecast")
st.write("Predict the total revenue (**Product_Store_Sales_Total**) a product will generate in a store. "
         f"Model: [`{MODEL_REPO}`](https://huggingface.co/{MODEL_REPO})")
model = load_model()

single, batch = st.tabs(["Single prediction", "Batch prediction (CSV)"])

with single:
    left, right = st.columns(2)
    with left:
        st.subheader("Product details")
        weight = st.number_input("Product weight", min_value=1.0, max_value=30.0, value=12.66, step=0.1)
        sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        area = st.slider("Allocated display area (ratio)", 0.0, 0.3, 0.068, step=0.001, format="%.3f")
        ptype = st.selectbox("Product type", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Frozen Foods"))
        mrp = st.number_input("Product MRP", min_value=1.0, max_value=500.0, value=147.0, step=1.0)
    with right:
        st.subheader("Store details")
        preset = st.selectbox("Store", list(STORES) + ["Custom / new store"], index=3)
        year, size, tier, stype = STORES.get(preset, (2005, "Medium", "Tier 2", "Supermarket Type1"))
        custom = preset not in STORES
        year = st.number_input("Establishment year", 1950, 2030, year, disabled=not custom)
        size = st.selectbox("Store size", SIZES, index=SIZES.index(size), disabled=not custom)
        tier = st.selectbox("City type", TIERS, index=TIERS.index(tier), disabled=not custom)
        stype = st.selectbox("Store type", STORE_TYPES, index=STORE_TYPES.index(stype), disabled=not custom)

    # Collect the inputs into a single-row DataFrame with the training column names
    input_df = pd.DataFrame([{
        "Product_Weight": weight, "Product_Sugar_Content": sugar, "Product_Allocated_Area": area,
        "Product_Type": ptype, "Product_MRP": mrp, "Store_Establishment_Year": int(year),
        "Store_Size": size, "Store_Location_City_Type": tier, "Store_Type": stype,
    }])
    st.dataframe(input_df, hide_index=True, use_container_width=True)

    if st.button("Predict sales", type="primary"):
        prediction = float(model.predict(input_df)[0])
        st.success(f"Forecast Product_Store_Sales_Total: **{prediction:,.2f}**")

with batch:
    st.write("Upload a CSV containing these columns: " + ", ".join(f"`{c}`" for c in FEATURES))
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["Product_Sugar_Content"] = data["Product_Sugar_Content"].replace({"reg": "Regular"})
            data["Predicted_Sales"] = model.predict(data[FEATURES]).round(2)
            st.metric("Total forecast revenue", f"{data['Predicted_Sales'].sum():,.2f}")
            st.dataframe(data, use_container_width=True)
            st.download_button("Download predictions", data.to_csv(index=False), "superkart_predictions.csv")
