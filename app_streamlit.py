import pandas as pd
import streamlit as st
import joblib

model = joblib.load("models_clafisicion_lemon.joblib")

st.set_page_config(
    page_title="Klasifikasi Lemon",
    page_icon=":lemon:"
)

st.title(":lemon: Klasifikasi Lemon")
st.markdown("Aplikasi Untuk Klasifikasi Kualitas Lemon")

diameter = st.slider("Diameter",45.5,68.5,50.5)
berat = st.slider("Berat",70.0,145.0,121.0)
tebal_kulit = st.slider("Tebal Kulit",3.4,6.0,4.2)
kadar_gula = st.slider("Kadar Gula",6.7,8.6,7.4)
asal_daerah = st.pills("Asal Daerah",["California","Malang","Medan"], default="Medan")
musim_panen = st.pills("Musim Panen",["Awal","Puncak","Akhir"], default="Awal")
warna = st.pills("Warna",["Hijau pekat","Kuning kehijauan","Kuning cerah"], default="Kuning cerah")

if st.button("Prediksi", type="primary"):
    data = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,musim_panen,warna]],
                   columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","musim_panen","warna"])

    predik = model.predict(data)[0]
    perserntase = max(model.predict_proba(data)[0])
    st.success(f"model memprediksi {predik} dengan persentase {perserntase*100:.2f}%")
    st.balloons()

st.divider()
st.caption("Dibuat dengan :lemon: oleh *Khairul faiz*)