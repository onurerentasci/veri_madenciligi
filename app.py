# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🌾 Tarımsal Ürün Tavsiye Sistemi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilizasyonu
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }    .metric-card {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 0.5rem 0;
        color: #2E7D32;
    }
    .recommendation-card {
        background-color: #E8F5E8;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
        text-align: center;
        color: #1B5E20;
    }
    .warning-card {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
        color: #E65100;
    }
    .info-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
        color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    try:
        with open('model/crop_recommendation_model_kfold.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model_info
    except FileNotFoundError:
        st.error("🚨 Model dosyası bulunamadı! Lütfen önce modeli eğittiğinizden emin olun.")
        return None

# Tahmin fonksiyonu
def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model_info):
    """
    Verilen toprak ve iklim koşullarına göre en uygun ürünü önerir.
    """
    try:
        model = model_info['model']
        scaler = model_info['scaler']
        le = model_info['label_encoder']
        
        # Girdi verilerini numpy dizisine dönüştür
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Verileri ölçeklendir
        input_scaled = scaler.transform(input_data)
        
        # Tahmin yap
        prediction = model.predict(input_scaled)[0]
        crop_name = le.inverse_transform([prediction])[0]
        
        # Sınıf olasılıklarını al
        probabilities = model.predict_proba(input_scaled)[0]
        
        # En yüksek olasılığa sahip 5 ürünü al
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_crops = le.inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        result = {
            'recommended_crop': crop_name,
            'confidence': round(np.max(probabilities) * 100, 2),
            'model_info': f"{model_info['model_name']} ({model_info.get('k_fold', 'N/A')}-fold CV ile optimize edilmiş)",
            'top_recommendations': [{'crop': crop, 'probability': round(prob * 100, 2)} 
                                  for crop, prob in zip(top_crops, top_probs)]
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

# Veri setini yükleme (grafikler için)
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('data/Crop_recommendation.csv')
        return df
    except FileNotFoundError:
        st.warning("⚠️ Veri seti dosyası bulunamadı. Bazı özellikler kullanılamayabilir.")
        return None

# Ana uygulama
def main():
    # Başlık
    st.markdown('<h1 class="main-header">🌾 Tarımsal Ürün Tavsiye Sistemi</h1>', unsafe_allow_html=True)
    
    # Model yükleme
    model_info = load_model()
    if model_info is None:
        st.stop()
    
    # Veri seti yükleme
    df = load_dataset()
      # Ana ekranda parametre girişi
    st.markdown("## 📊 Toprak ve İklim Parametreleri")
    st.markdown("Aşağıdaki değerleri girerek tarlanız için en uygun ürün önerisini alın:")
    
    # Parametre girişi için üç sütun
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        st.markdown("### 🌱 Toprak Besin Elementleri")
        N = st.slider(
            "Azot (N) - kg/ha", 
            min_value=0, max_value=150, value=90, step=1,
            help="Topraktaki azot miktarı (kg/hektar)"
        )
        
        P = st.slider(
            "Fosfor (P) - kg/ha", 
            min_value=5, max_value=150, value=42, step=1,
            help="Topraktaki fosfor miktarı (kg/hektar)"
        )
        
        K = st.slider(
            "Potasyum (K) - kg/ha", 
            min_value=5, max_value=210, value=43, step=1,
            help="Topraktaki potasyum miktarı (kg/hektar)"
        )
    
    with param_col2:
        st.markdown("### 🌤️ İklim Koşulları")
        temperature = st.slider(
            "Sıcaklık - °C", 
            min_value=8.0, max_value=45.0, value=25.0, step=0.1,
            help="Ortalama hava sıcaklığı (Celsius)"
        )
        
        humidity = st.slider(
            "Nem - %", 
            min_value=14.0, max_value=100.0, value=80.0, step=0.1,
            help="Ortalama bağıl nem yüzdesi"
        )
        
        rainfall = st.slider(
            "Yağış - mm", 
            min_value=20.0, max_value=300.0, value=200.0, step=1.0,
            help="Yıllık yağış miktarı (milimetre)"
        )
    
    with param_col3:
        st.markdown("### 🧪 Toprak Özellikleri")
        ph = st.slider(
            "pH Değeri", 
            min_value=3.5, max_value=10.0, value=6.5, step=0.1,
            help="Toprak pH değeri (3.5-10.0 arası)"
        )
        
        # Tahmin butonu
        st.markdown("### 🎯 Tahmin")
        if st.button("🌾 Ürün Önerisi Al", type="primary", use_container_width=True):
            with st.spinner('🔄 Model çalışıyor...'):
                result = predict_crop(N, P, K, temperature, humidity, ph, rainfall, model_info)
                
                if 'error' in result:
                    st.error(f"❌ Hata: {result['error']}")
                else:
                    # Sonuçları session state'e kaydet
                    st.session_state['prediction_result'] = result
                    st.session_state['input_values'] = {
                        'N': N, 'P': P, 'K': K, 'temperature': temperature,
                        'humidity': humidity, 'ph': ph, 'rainfall': rainfall
                    }
    
    st.markdown("---")
      # Sidebar - Bilgi kartları
    st.sidebar.markdown("## ℹ️ Model Bilgileri")
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Model Türü:</strong> {model_info['model_name']}<br>
        <strong>Optimizasyon:</strong> {model_info.get('k_fold', 'N/A')}-fold Çapraz Doğrulama<br>
        <strong>Özellik Sayısı:</strong> {len(model_info['features'])}<br>
        <strong>Sınıf Sayısı:</strong> {len(model_info['label_encoder'].classes_)}
    </div>
    """, unsafe_allow_html=True)
    
    # Mevcut girdi değerleri
    if 'input_values' in st.session_state:
        st.sidebar.markdown("## 📊 Girilen Değerler")
        input_vals = st.session_state['input_values']
        
        st.sidebar.markdown(f"""
        <div class="info-card">
            <strong>Azot (N):</strong> {input_vals['N']} kg/ha<br>
            <strong>Fosfor (P):</strong> {input_vals['P']} kg/ha<br>
            <strong>Potasyum (K):</strong> {input_vals['K']} kg/ha<br>
            <strong>Sıcaklık:</strong> {input_vals['temperature']} °C<br>
            <strong>Nem:</strong> {input_vals['humidity']} %<br>
            <strong>pH:</strong> {input_vals['ph']}<br>
            <strong>Yağış:</strong> {input_vals['rainfall']} mm
        </div>
        """, unsafe_allow_html=True)
    
    # Veri seti genel bilgileri
    if df is not None:
        st.sidebar.markdown("## 📈 Veri Seti Özeti")
        
        st.sidebar.markdown(f"""
        <div class="warning-card">
            <strong>Toplam Örnek:</strong> {len(df)}<br>
            <strong>Ürün Türü:</strong> {df['label'].nunique()}<br>
            <strong>En Yaygın Ürün:</strong> {df['label'].value_counts().index[0]}<br>
            <strong>Parametreler:</strong> N, P, K, Sıcaklık, Nem, pH, Yağış
        </div>
        """, unsafe_allow_html=True)
      # Ana içerik alanı - sadece grafikler ve sonuçlar
    main_col1, main_col2 = st.columns([3, 2])
    
    with main_col1:
        # Tahmin sonuçları
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            
            st.markdown('<h2 class="sub-header">🎯 Ürün Önerisi Sonucu</h2>', unsafe_allow_html=True)
              # Ana öneri kartı
            st.markdown(f"""
            <div class="recommendation-card">
                <h2 style="color: #1B5E20;">🌾 Önerilen Ürün</h2>
                <h1 style="color: #2E7D32; margin: 1rem 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">{result['recommended_crop'].upper()}</h1>
                <h3 style="color: #388E3C;">Güven Oranı: %{result['confidence']}</h3>
                <p style="color: #4CAF50; margin-top: 1rem; font-weight: 500;">Model: {result['model_info']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # En iyi 5 öneri
            st.markdown('<h3 class="sub-header">📈 Tüm Öneriler (Olasılık Sıralaması)</h3>', unsafe_allow_html=True)
              # Grafik oluşturma
            recommendations_df = pd.DataFrame(result['top_recommendations'])
            
            fig = px.bar(
                recommendations_df, 
                x='probability', 
                y='crop',
                orientation='h',
                title="Ürün Önerileri ve Olasılıkları (%)",
                labels={'probability': 'Olasılık (%)', 'crop': 'Ürün'},
                color='probability',
                color_continuous_scale='Viridis',
                text='probability'
            )
            
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig.update_layout(
                height=450,
                showlegend=False,
                title_font_size=16,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Olasılık (%)",
                yaxis_title="Ürün Türü",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tablo görünümü
            st.markdown("### 📋 Detaylı Sonuçlar")
            recommendations_df['Sıra'] = range(1, len(recommendations_df) + 1)
            recommendations_df = recommendations_df[['Sıra', 'crop', 'probability']].rename(columns={
                'crop': 'Ürün',
                'probability': 'Olasılık (%)'
            })
            
            st.dataframe(
                recommendations_df,                use_container_width=True,
                hide_index=True
            )
        
        else:
            # Başlangıç mesajı
            st.markdown("""
            <div class="warning-card">
                <h3 style="color: #E65100;">👋 Hoş Geldiniz!</h3>
                <p style="color: #F57C00;">Toprak ve iklim verilerinizi yukarıdaki parametrelerden girerek tarlanız için en uygun ürün önerisini alabilirsiniz.</p>
                <ul style="color: #FF9800;">
                    <li>🌱 Toprak besin elementlerini (N, P, K) girin</li>
                    <li>🌤️ İklim koşullarını (sıcaklık, nem, yağış) belirtin</li>
                    <li>🧪 Toprak pH değerini ayarlayın</li>
                    <li>🎯 "Ürün Önerisi Al" butonuna basın</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with main_col2:
        # Veri seti istatistikleri (eğer mevcut ise) - Sadece grafikler
        if df is not None:
            st.markdown('<h3 class="sub-header">📈 Veri Seti Grafikleri</h3>', unsafe_allow_html=True)
            
            # Ürün dağılımı pasta grafiği
            crop_counts = df['label'].value_counts().head(8)
            
            fig_pie = px.pie(
                values=crop_counts.values,
                names=crop_counts.index,
                title="En Yaygın 8 Ürünün Dağılımı",
                color_discrete_sequence=px.colors.qualitative.Set3            )
            
            fig_pie.update_layout(
                height=400, 
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                title_font_size=16,
                margin=dict(l=20, r=20, t=40, b=80)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Besin elementi dağılımları
            st.markdown("### 🧪 Besin Elementi Analizi")
              # NPK dağılımı histogram - daha kompakt
            fig_nutrients = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Azot (N)', 'Fosfor (P)', 'Potasyum (K)'),
                vertical_spacing=0.08
            )
            
            fig_nutrients.add_trace(
                go.Histogram(x=df['N'], name='Azot', marker_color='lightgreen', opacity=0.8, nbinsx=15),
                row=1, col=1
            )
            fig_nutrients.add_trace(
                go.Histogram(x=df['P'], name='Fosfor', marker_color='lightcoral', opacity=0.8, nbinsx=15),
                row=2, col=1
            )
            fig_nutrients.add_trace(
                go.Histogram(x=df['K'], name='Potasyum', marker_color='lightblue', opacity=0.8, nbinsx=15),
                row=3, col=1
            )
            
            fig_nutrients.update_layout(
                title_text="Besin Elementleri Dağılımı",
                height=500,
                showlegend=False,
                title_font_size=16,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig_nutrients, use_container_width=True)
              # İklim koşulları analizi
            st.markdown("### 🌡️ İklim Koşulları Analizi")
            
            fig_climate = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sıcaklık', 'Nem', 'pH', 'Yağış'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            fig_climate.add_trace(
                go.Box(y=df['temperature'], name='Sıcaklık', marker_color='orange', showlegend=False),
                row=1, col=1
            )
            fig_climate.add_trace(
                go.Box(y=df['humidity'], name='Nem', marker_color='blue', showlegend=False),
                row=1, col=2
            )
            fig_climate.add_trace(
                go.Box(y=df['ph'], name='pH', marker_color='purple', showlegend=False),
                row=2, col=1
            )
            fig_climate.add_trace(
                go.Box(y=df['rainfall'], name='Yağış', marker_color='teal', showlegend=False),
                row=2, col=2
            )
            
            fig_climate.update_layout(
                title_text="İklim Parametreleri Analizi",
                height=450,
                showlegend=False,
                title_font_size=16,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig_climate, use_container_width=True)
              # Korelasyon matrisi
            st.markdown("### 🔗 Parametre İlişkileri")
            
            correlation_matrix = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Parametreler Arası Korelasyon Matrisi",
                color_continuous_scale='RdBu',
                aspect="auto",
                text_auto=True
            )
            
            fig_corr.update_layout(
                height=450,
                title_font_size=16,
                margin=dict(l=60, r=60, t=60, b=60)
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🌾 Tarımsal Ürün Tavsiye Sistemi | Machine Learning ile Geliştirilmiştir</p>
        <p><small>Bu sistem, toprak ve iklim verilerini analiz ederek en uygun ürün önerisini sunar.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
