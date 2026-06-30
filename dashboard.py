import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")
st.image("tbt_coverkpop_header.jpg")
st.markdown("""
<style>
/* 🔹 Main page background */
.stApp {
    background-color: white;
}

/* 🔹 Main title (st.title) */
.block-container  {
    padding-top: 80px;
    }
h1 {
    background: linear-gradient(90deg, #0D47A1, #424242);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        text-align: center;
}

/* 🔹 Section titles (st.subheader) */
h2, h3 {
    color: black !important;
    font-size:20px;
}

/*tab titles*/
button[data-baseweb="tab"] {
    background-color: white !important;
    font-weight: bold;
    color: black !important;
    font-size: 20px;
}
/* 🔹 Sidebar background */
section[data-testid="stSidebar"] {
    background-color: white !important;
}

/* 🔹 Sidebar title */
section[data-testid="stSidebar"] h1 {
    color: black !important;
    font-size:25px;
}

/* 🔹 Sidebar labels */
section[data-testid="stSidebar"] label {
    color: black !important;
    font-weight: 500;
}

/* 🔹 Sidebar inputs text */
section[data-testid="stSidebar"] .stSelectbox div,
section[data-testid="stSidebar"] .stDateInput div,
section[data-testid="stSidebar"] .stMultiSelect div {
    color: black !important;
}
/* 🔹 General text */
body, p, span {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)
st.title("🎧 K-Pop Comeback Momentum & Fandom Analysis")
df=pd.read_csv("Atlantic_South_Korea.csv")

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Duration_ms'] = pd.to_numeric(df['Duration_ms'], errors='coerce')
df['duration_min'] = df['Duration_ms'] / 60000
df = df.drop_duplicates(['Artist', 'Song', 'Date'])
df = df.sort_values(['Artist', 'Date'])

st.sidebar.image("logo.png", width=180)

min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

selected_artists = st.sidebar.multiselect("Select Artists",options=sorted(df['Artist'].dropna().unique()))
selected_songs = st.sidebar.multiselect("Select Songs",options=sorted(df['Song'].dropna().unique()))
album_options = df['Album_type'].dropna().unique().isin(['album', 'single'])
selected_album_types = st.sidebar.multiselect("Select Album Types",options=['album', 'single'])



# Re-entry
df['song_id'] = df['Song'] + " - " + df['Artist']
df['prev_date'] = df.groupby('song_id')['Date'].shift(1)
df['gap_days'] = (df['Date'] - df['prev_date']).dt.days

EXPECTED_GAP = 7
df['is_reentry'] = df['gap_days'] > EXPECTED_GAP
df.loc[df['prev_date'].isna(), 'is_reentry'] = False

reentry_metrics = (
    df.groupby('Artist')
    .agg(
        total_reentries=('is_reentry', 'sum'),
        total_appearances=('Date', 'count')
    )
    .reset_index()
)
reentry_metrics['reentry_frequency'] = (
    reentry_metrics['total_reentries'] /
    reentry_metrics['total_appearances']
)
df = df.merge(reentry_metrics, on='Artist', how='left')
reentry_range = st.sidebar.slider(
    "Re-entry Count Range",
    min_value=0,
    max_value=int(df['total_reentries'].max()),
    value=(0, int(df['total_reentries'].max()))
)

#momentum spike
df['prev_Position'] = df.groupby(['song_id'])['Position'].shift(1)
df['Position_change'] = df['prev_Position'] - df['Position']
df['momentum_spike'] = df['Position_change'].clip(lower=0)

momentum_metrics = (
    df.groupby('song_id')
    .agg(
        total_spike=('momentum_spike', 'sum'),
        avg_spike=('momentum_spike', 'mean'),
        max_spike=('momentum_spike', 'max')
    )
    .reset_index()
)
df = df.merge(momentum_metrics, on='song_id', how='left')


#Retention Analysis 

EXPECTED_GAP = 7 
df = df.sort_values(['Artist', 'Song', 'Date'])
df['gap_days'] = df.groupby(['Artist', 'Song'])['Date'].diff().dt.days
df['new_segment'] = df['gap_days'] > EXPECTED_GAP
df['segment_id'] = df.groupby(['Artist', 'Song'])['new_segment'].cumsum()
retention = (
    df.groupby(['Artist', 'Song', 'segment_id'])
    .agg(
        start_date=('Date', 'min'),
        end_date=('Date', 'max'),
        days_on_chart=('Date', 'count')
    )
    .reset_index()
)
df['is_segment_start'] = df.groupby(['Artist', 'Song', 'segment_id']).cumcount() == 0
reentry_segments = df[
    (df['is_segment_start']) & (df['is_reentry'])
][['Artist', 'Song', 'segment_id']]
retention = retention.merge(
    reentry_segments,
    on=['Artist', 'Song', 'segment_id'],
    how='inner'
)

retention_metrics = (
    retention.groupby('Artist')
    .agg(
        avg_retention=('days_on_chart', 'mean'),
        max_retention=('days_on_chart', 'max')
    )
    .reset_index()
)
df = df.merge(retention_metrics, on='Artist', how='left')

# recovery speed calculation
df['next_position'] = df.groupby(['Artist','Song'])['Position'].shift(-1)
df['next_date'] = df.groupby(['Artist','Song'])['Date'].shift(-1)
df['position_improvement'] = df['Position'] - df['next_position']
df['days_taken'] = (df['next_date'] - df['Date']).dt.days
df['recovery_speed'] = 0.0

mask = df['is_reentry'] & (df['days_taken'] > 0) & (df['position_improvement'] > 0)

df.loc[mask, 'recovery_speed'] = (
    (df.loc[mask, 'position_improvement'] /
    df.loc[mask, 'days_taken'])*100
)
df['recovery_speed'] = df['recovery_speed'].clip(lower=0)
recovery_metrics = (
    df.groupby(['Artist','Song'])
    .agg(
        avg_recovery_speed=('recovery_speed', 'mean'),
        max_recovery_speed=('recovery_speed', 'max')
    )
    .reset_index()
)
df = df.merge(recovery_metrics, on=['Artist','Song'], how='left')

# Album comeback advantage index (ACAI) calculation
df['Album_type'] = df['Album_type'].str.strip().str.lower()
reentry_df = df[df['is_reentry'] == True]
album_metrics = (
    reentry_df.groupby('Album_type')
    .agg(avg_recovery_speed=('recovery_speed', 'mean'))
)

album_speed = album_metrics['avg_recovery_speed'].get('album', 0)
single_speed = album_metrics['avg_recovery_speed'].get('single', 0)
acai = album_speed - single_speed
delta = "Album > Single" if acai > 0 else "Single > Album"

#Fandom Intensity Proxy Score
def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)
df['norm_freq'] = norm(df['reentry_frequency'])
df['norm_retention'] = norm(df['avg_retention'])
df['norm_recovery'] = norm(df['avg_recovery_speed'])
df['fandom_intensity_score'] = (
    0.3 * df['norm_freq'] +
    0.4 * df['norm_retention'] +
    0.3 * df['norm_recovery']
)

#Comeback vs First Entry Performance
df['entry_number'] = df.groupby('Artist')['is_reentry'].cumsum() + 1
df['entry_type'] = df['entry_number'].apply(
    lambda x: 'First Entry' if x == 1 else 'Re-entry'
)

#album type distribution for visualization
album_dist = (
    df['Album_type']
    .value_counts()
    .reset_index()
)

album_dist.columns = ['Album_type', 'count']

filtered_df = df.copy()

if selected_artists:
    filtered_df = filtered_df[filtered_df['Artist'].isin(selected_artists)]

if selected_songs:
    filtered_df = filtered_df[filtered_df['Song'].isin(selected_songs)]

if len(date_range) == 2:
    start_date, end_date = date_range

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = filtered_df[
        (filtered_df['Date'] >= start_date) &
        (filtered_df['Date'] <= end_date)
    ]

if selected_album_types:
    filtered_df = filtered_df[filtered_df['Album_type'].isin(selected_album_types)]

    
min_val, max_val = reentry_range
filtered_df = filtered_df[
    (filtered_df['total_reentries'] >= min_val) &
    (filtered_df['total_reentries'] <= max_val)
]

artist_counts = filtered_df.groupby('Artist')['Song'].nunique().reset_index()
artist_counts.columns = ['Artist', 'song_count']

#top songs by momentum spike and re-entry count
top_songs = filtered_df.groupby(['Song', 'Artist']).agg({
    'momentum_spike': 'mean',
    'is_reentry': 'sum',
    'recovery_speed': 'mean'
}).reset_index()
top_songs['reentries'] = (top_songs['is_reentry'] - 1).clip(lower=0)
top_songs['score'] = (
    top_songs['momentum_spike'] * 0.6 +
    top_songs['reentries'] * 0.4 +top_songs['recovery_speed'] * 0.3
)
top_songs = top_songs.sort_values('score', ascending=False).head(6)
image_df = (
    filtered_df
    .sort_values('Date')  
    .drop_duplicates(subset=['Song', 'Artist'])
    [['Song', 'Artist', 'Album_cover_url']]
)

top_songs = top_songs.merge(
    image_df,
    on=['Song', 'Artist'],
    how='left'
)

#kpi card function
def kpi_card(title, value, icon=None, is_positive=True):
    if is_positive:
        gradient = "linear-gradient(135deg, #7c3aed, #a78bfa)"
    else:
        gradient = "linear-gradient(135deg, #ef4444, #f87171)"

    st.markdown(f"""
     <div style="
        width:100%;
        background:green;
        padding:15px;
        border-radius:12px;
        color:white;
        margin-bottom:20px;
        ">
        <div style="font-size:26px; font-weight:600;margin-bottom:5px;">
            {title}
        </div>

        <div style="font-size:28px; font-weight:bold; margin-top:5px;">
            {value}
        </div>
     </div>
     """, unsafe_allow_html=True)
    
col1, col2, col3= st.columns(3)
def safe_mean(series):
    return round(series.mean(), 2) if not series.dropna().empty else 0
with col1:
    kpi_card("Recovery Speed", round(filtered_df['avg_recovery_speed'].mean(),2),icon="⚡")
with col2:
    kpi_card("Fandom Intensity", safe_mean(filtered_df['fandom_intensity_score']),icon="🔥")
with col3:
    kpi_card("Avg Retention", safe_mean(filtered_df['avg_retention']),icon="📊",is_positive=True)


col4, col5, col6 = st.columns(3)
with col4:
    kpi_card("Average Re-entry Frequency",filtered_df['reentry_frequency'].mean().round(2),icon="🔁")
with col5:
    is_positive = acai > 0
    kpi_card(
        title="Album Comeback Index",
        value=round(acai, 2),
        icon="💿",
        is_positive=is_positive
    )
with col6:
    kpi_card("Avg Momentum Spike Score", round(filtered_df['avg_spike'].mean(), 2),icon="⚡")


tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs([
    "**📊 Overview**",
    "**🔥 Fandom Intensity Leaderboard**",
    "**📈 Momentum Spike Analysis**",
    "**🔁 Re-entry Status**",
    "**📊 Retention Analysis**",
    "**📂 Data Preview**"
])
with tab1:
    col_left, col_right = st.columns([1,2])
    with col_left:
        st.subheader("🎵 Top Songs")

        for _, row in top_songs.iterrows():
            c1, c2 = st.columns([1, 4])
            with c1:
                st.image(row['Album_cover_url'], width=60)  

            with c2:
                st.markdown(f"**{row['Song']}**")
                st.markdown(f"*{row['Artist']}*")

    with col_right:
        st.subheader("🎤 Top Artists")
        top_artists = filtered_df.drop_duplicates('Artist').sort_values('fandom_intensity_score', ascending=False).head(15)
        fig = px.bar(
        top_artists,
        x='fandom_intensity_score',
        y='Artist',
        orientation='h',
        color='fandom_intensity_score',
        color_continuous_scale="Viridis",
        )
        fig.update_layout(
            xaxis_title="Fandom Intensity Score",
            yaxis_title="Artist",
            height=500,
            margin=dict(l=10, r=10, t=0, b=10),
            yaxis=dict(autorange="reversed", automargin=True),
            coloraxis_colorbar=dict(title="Fandom IntensityScore"),
        )
        fig.update_xaxes(title_font_size=16,title_font_color="black",tickfont_size=10,tickfont_color="black")
        fig.update_yaxes(title_font_size=16,title_font_color="black",tickfont_size=14,tickfont_color="black")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <p style="
    font-size:19px;
    color:#e5e7eb;
    margin-top:0;
    ">
    The top artists are ranked by their Fandom Intensity Score, which combines re-entry frequency
    retention, and recovery speed to reflect the overall engagement and loyalty of their fanbase. Higher scores indicate stronger, more active fandoms that drive sustained success on the charts.
    </p>
    """, unsafe_allow_html=True)
with tab2:
    st.subheader("🏆 Fandom Intensity Leaderboard")
    st.markdown("""
    <p style="
    font-size:19px;
    color:#e5e7eb;
    margin-top:0;
    ">
    The Fandom Intensity Score combines re-entry frequency, retention, and recovery speed to rank artists based on their fanbase's engagement and loyalty. Higher scores indicate stronger, more active fandoms that drive sustained success on the charts.
    </p>
    """, unsafe_allow_html=True)
    leaderboard_df = (
    filtered_df.drop_duplicates('Artist').sort_values('fandom_intensity_score', ascending=False)).reset_index(drop=True)
    min_n = 1
    max_n = len(leaderboard_df)
    if max_n <= 1:
        top_n = 1
    else:
        top_n = st.slider(
        "Select Top N Artists",
        min_value=1,
        max_value=max_n,
        value=min(10, max_n)
        )
    top_artists = leaderboard_df.head(top_n)
    st.dataframe(
    top_artists[['Artist', 'fandom_intensity_score']].style.set_properties(**{
        'font-weight':'bold',
        'background-color': 'white',
        'color': 'black'}),use_container_width=True)
with tab3:
    st.subheader("📈 Momentum Spike Analysis")
    avg_spike = filtered_df['avg_spike'].mean()
    top_artist = (filtered_df.drop_duplicates('Artist').sort_values('avg_spike', ascending=False).iloc[0]['Artist'])
    top_song = (filtered_df.drop_duplicates('Song').sort_values('avg_spike', ascending=False).iloc[0]['Song'])
    
    st.markdown("<p style='font-size:19px; color:#e5e7eb;'>➤Select multiple album types to compare their average momentum spikes</p>", unsafe_allow_html=True)
    album_pref = filtered_df.groupby('Album_type').agg(avg_spike=('avg_spike', 'mean'))
    album_value= album_pref['avg_spike'].get('album', 0)
    single_value = album_pref['avg_spike'].get('single', 0)
    st.write('Selected Album Momentum Spike Score:',round(album_value, 2))
    st.write('Selected Single Momentum Spike Score:',round(single_value, 2))

    if album_value > single_value:
        album_text = "Albums are generating stronger momentum spikes than singles."
    elif single_value > album_value:
        album_text = "Singles are generating stronger momentum spikes than albums."
    else:
        album_text = "Singles and albums are generating similar momentum spikes."
    insight_text = f"""
        The average momentum spike is <b><span style='color:#60a5fa'>{round(avg_spike, 2)}</span></b>, indicating overall <b><span style='color:#60a5fa'>{('high' if avg_spike > 5 else 'moderate' if avg_spike > 2 else 'low')}</span></b> volatility in chart movement
        with <b><span style='color:#60a5fa'>{top_artist}</span></b> showing the strongest artist-level momentum, suggesting highly effective release engagement.
        The song <b><span style='color:#60a5fa'>{top_song}</span></b> demonstrates the highest spike impact among all tracks.
        <b><span style='color:#60a5fa'>{album_text}</span></b>"""
    st.markdown(f"<p style='font-size:25px; color:#e5e7eb;'>{insight_text}</p>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1,1])
    with col_left:
        fig2 = px.line(
        filtered_df.drop_duplicates('Artist'),
        x='Artist',
        y='avg_spike',
        title="Average Momentum Spike Score by Artist",
        color='Album_type',
        color_discrete_map={
            'album': "#21be3b",
            'single': "#930eff"
        }
        )
        fig2.update_layout(
        xaxis_title="Artist",
        yaxis_title="Avg Momentum Spike",
        xaxis_title_font=dict(size=14, color="#00040c"),
        yaxis_title_font=dict(size=14, color="#00040a"),
        legend=dict(title="Album Type", title_font=dict(size=12, color="#01050f"),font=dict(color="#00050f", size=12)),
        font=dict(color="#0f0000"),
        title_font=dict(size=16, color="#0e0000"),
        height=500,
        title_x=0.1
        )
        
        fig2.update_xaxes(
        tickfont=dict(color="#01060f", size=10)
        )
        fig2.update_yaxes(
        tickfont=dict(color="#000308", size=10),
        zeroline=False,
        showgrid=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_right:
        fig3 = px.line(
        filtered_df.drop_duplicates('Song'),
        x='Song',
        y='avg_spike',
        title="Average Momentum Spike Score by Song",
        color='Album_type',
        color_discrete_map={
            'album': "#21be3b",
            'single': "#930eff"
        }
        )
        fig3.update_layout(
        xaxis_title="Song",
        yaxis_title="Avg Momentum Spike",
        xaxis_title_font=dict(size=14, color="#01050e"),
        yaxis_title_font=dict(size=14, color="#00040c"),
        legend=dict(title="Album Type", title_font=dict(size=12, color="#00050e"),font=dict(color="#000611", size=12)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#000611"),
        title_font=dict(size=16, color="#0a0000"),
        height=500,
        title_x=0.1
        )
        fig3.update_xaxes(
        tickfont=dict(color="#000308", size=10),
        showgrid=False
        )
        fig3.update_yaxes(
        tickfont=dict(color="#00040c", size=10),
        showgrid=False,
        zeroline=False
        
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    filtered_df['Album_type'] = (filtered_df['Album_type'].str.strip().str.title())
    
    album_comparison_df = (
        filtered_df.groupby('Album_type')
        .agg(
        avg_spike=('avg_spike', 'mean'),
        avg_recovery=('recovery_speed', 'mean'),
        avg_retention=('avg_retention', 'mean'),
        avg_intensity=('fandom_intensity_score', 'mean')
        )
        .reset_index()
    )
    col_left, col_right = st.columns([2,3])
    with col_left:
        fig1 = px.bar(
        album_comparison_df,
        x='Album_type',
        y='avg_spike',
        color='Album_type',
        color_discrete_map={
            'Album': "#21be3b",
            'Single': "#930eff"},
        title="Momentum Spike"
        )
        fig1.update_layout(
        xaxis_title="Album Type",
        yaxis_title="Avg Momentum Spike",
        height=500,   
        font=dict(color="#e5e7eb"), 
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        title_x=0.2
        )
        fig1.update_xaxes(
        tickfont=dict(color="#00050f", size=14),
        zeroline=False,title_font_color="black"
        )
        fig1.update_yaxes(
        tickfont=dict(color="#00040a", size=14),title_font_color="black",
        showgrid=False
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col_right:
        fig = px.scatter(
        filtered_df,
        x='duration_min',
        y='avg_spike',
        color='Album_type',
        color_discrete_map={
            'Album': "#21be3b",
            'Single': "#930eff"
        },
     )
        fig.update_layout(
        title={
        'text': "Duration vs Momentum Spike",
        'x': 0.2
        },
        xaxis_title="Duration (min)",
        yaxis_title="Avg Momentum Spike",    
        font=dict(color="#00040c"),  
        margin=dict(l=10, r=10, t=40, b=10),height=500,
        legend_title="Album Type",
        legend=dict(title="Album Type", title_font=dict(size=12, color="#00050e"),font=dict(color="#000611", size=12)),
        )
        fig.update_xaxes(
        tickfont=dict(color="#00040a", size=14),
        showgrid=False
        )
        fig.update_yaxes(
        tickfont=dict(color="#00030a", size=14)
        )
        st.plotly_chart(fig, use_container_width=True)      
with tab4:
    fig1 = px.bar(
        filtered_df.drop_duplicates('Artist').sort_values('reentry_frequency', ascending=False).head(),
        x='Artist',
        y='reentry_frequency',
        title="Fandom Reactivation",
        color='Album_type',
        color_discrete_map={
            'Album': "#21be3b",
            'Single': "#930eff"
            }
        )
    fig1.update_layout(
        title_font=dict(size=17, color="#010611"),
        xaxis_title="Artist",
        yaxis_title="Re-entry Frequency",
        font=dict(color= "#000308"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(title="Album Type", title_font=dict(size=12, color="#00050e"),font=dict(color="#000611", size=12))
        )
    fig.update_xaxes(
        tickfont=dict(color="#00040a", size=14),
        showgrid=False
        )
    fig.update_yaxes(
        tickfont=dict(color="#00030a", size=14)
        )
    
    st.plotly_chart(fig1,use_container_width=True)
    
    col_left, col_right = st.columns(2)

    with col_left:
        fig1 = px.bar(
        filtered_df,
        x='entry_type',
        y='duration_min',
        color='entry_type',
        color_discrete_map={
        'First Entry': "#4eb96f",
        'Re-entry': "#19A599"
        }
        )
        fig1.update_layout(
            title={
            'text': "Comeback vs First Entry Performance",
            'x': 0.2
            },
            xaxis_title="Entry Type",
            yaxis_title="Duration (min)",
            font=dict(color="#000308"),  
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        fig1.update_yaxes(
            showgrid=False
        )

        st.plotly_chart(fig1, use_container_width=True)
        
    entry_comparison_df = (
    df.groupby('entry_type')
    .agg(
        avg_position=('Position', 'mean')
    )
    .reset_index()
        )
    
    with col_right:
        fig1 = px.bar(
        entry_comparison_df,
        x='entry_type',
        y='avg_position',
        title="Comeback vs First Entry Performance",
        color='entry_type',
        color_discrete_map={
        'First Entry': "#4eb96f",
        'Re-entry': "#19A599"
        }
        )
        fig1.update_layout(
        title={
        'text': "Comeback vs First Entry Performance",
        'x': 0.2
        },
        xaxis_title="Entry Type",
        yaxis_title="Avg Chart Position",
        font=dict(color="#00040c"), 
        margin=dict(l=10, r=10, t=40, b=10)
        )
        fig1.update_yaxes(
            showgrid=False
        )

        st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    <p style="
    font-size:19px;
    color:#e5e7eb;
    margin-top:0;
    ">
    The Fandom Intensity Score combines re-entry frequency, retention, and recovery speed to rank artists based on their fanbase's engagement and loyalty. Higher scores indicate stronger, more active fandoms that drive sustained success on the charts.
    </p>
    """, unsafe_allow_html=True)
with tab5:
    
    avg_ret = retention_metrics['avg_retention'].mean()

    if avg_ret > 6:
        status = "- 🔥 Strong Retention"
    elif avg_ret > 3:
        status = "- ⚡ Moderate Retention"
    else:
     status = "- 📉 Weak Retention"

    st.markdown(f"""
        <h3>📊 Retention Analysis  {status}</h3>
        """, unsafe_allow_html=True)
    top_retention=filtered_df.drop_duplicates('Artist').sort_values('avg_retention', ascending=False).head(35)
    fig1 = px.bar(
        top_retention,
        x='Artist',
        y='avg_retention',
        title="Avg Retention by Artist",
        color='avg_retention',
        color_continuous_scale="Viridis"
    )
    fig1.update_layout(
        title_font=dict(size=17, color="#000308"),
        xaxis_title="Artist",
        yaxis_title="Average Retention",
        font=dict(color="#000308"), 
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(title="Avg Retention"),
        coloraxis_showscale=False,
        
    )
    fig1.update_xaxes(
        tickangle=45
    )
    fig1.update_yaxes(
        showgrid=False
    )
    st.plotly_chart(fig1,use_container_width=True)
    
with tab6:
    st.markdown("## 📂 Dataset Explorer")
    search = st.text_input("🔍 Search Artist or Song")
    df_display = filtered_df.copy()
    if search:
        df_display = df_display[
            df_display['Artist'].str.contains(search, case=False, na=False) |
            df_display['Song'].str.contains(search, case=False, na=False)
        ]
    
    st.dataframe(df_display, use_container_width=True)
    st.download_button(
        "⬇ Download CSV",
        df_display.to_csv(index=False),
        file_name="filtered_data.csv"
    )
