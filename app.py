"""
app.py — Nordic Energy Price Forecast Dashboard (punto de entrada)
==================================================================
Uso:  streamlit run app.py

Define la navegación de la barra lateral con st.navigation. La portada
(pages/0_Overview.py) aparece como «Overview» y se sirve en la URL raíz; las
otras seis páginas conservan el título y la URL que Streamlit deduce del
nombre del fichero (/Market_Analysis, /Detailed_Forecast, /Value_Calculator,
/Technical_Docs, /Battery_Lab, /Transparency).
"""

import streamlit as st

navigation = st.navigation([
    st.Page("pages/0_Overview.py", title="Overview", default=True),
    st.Page("pages/1_Market_Analysis.py"),
    st.Page("pages/2_Detailed_Forecast.py"),
    st.Page("pages/3_Value_Calculator.py"),
    st.Page("pages/4_Technical_Docs.py"),
    st.Page("pages/5_Battery_Lab.py"),
    st.Page("pages/6_Transparency.py"),
])
navigation.run()
