import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Music Marketing Analytics",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
/* Make expander titles bigger */
details summary {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    padding: 1rem !important;
    font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_data():
    platform_summary = pd.read_csv(BASE_DIR / "data" / "platform_summary.csv")
    campaign_summary = pd.read_csv(BASE_DIR / "data" / "campaign_summary.csv")
    daily_streams = pd.read_csv(BASE_DIR / "data" / "daily_streams_by_source.csv")

    # Convert dates
    date_cols = ['campaign_start_date', 'campaign_end_date', 'baseline_start', 
                 'baseline_end', 'during_start', 'during_end', 'decay_start', 'decay_end']
    for col in date_cols:
        if col in campaign_summary.columns:
            campaign_summary[col] = pd.to_datetime(campaign_summary[col], errors='coerce')
    
    daily_streams['report_date'] = pd.to_datetime(daily_streams['report_date'], errors='coerce')
    
    return platform_summary, campaign_summary, daily_streams

platform_summary, campaign_summary, daily_streams = load_data()

# ============================================
# COLORS
# ============================================

platform_colors_base = {
    'facebook': '#1877F2',
    'google': '#4285F4',
    'instagram': '#E4405F',
    'snapchat': '#FFFC00',
    'tiktok': '#000000',
    'youtube': '#FF0000'
}

platform_colors_bg = {
    'facebook': 'rgba(24, 119, 242, 0.25)',
    'instagram': 'rgba(228, 64, 95, 0.25)',
    'tiktok': 'rgba(0, 0, 0, 0.25)',
    'youtube': 'rgba(255, 0, 0, 0.25)',
    'snapchat': 'rgba(255, 252, 0, 0.25)',
    'google': 'rgba(66, 133, 244, 0.25)'
}

source_colors = {
    'search': '#FF69B4',
    'radio': '#90EE90',
    'play_queue': '#FF6B6B',
    'others playlist': '#87CEEB',
    'other': '#FFB366',
    'collection': '#DDA0DD',
    'chart': '#66CDAA',
    'artist': '#F08080',
    'album': '#9370DB'
}

confidence_alpha = {
    'A_reliable': 1.0,
    'B_indicative': 0.7,
    'C_exploratory': 0.4
}

platform_abbrev = {
    'facebook': 'FB', 'instagram': 'IG', 'tiktok': 'TT',
    'youtube': 'YT', 'snapchat': 'SC', 'google': 'GG'
}

# ============================================
# HEADER
# ============================================

# 1. hollyowl.jpg - banner at the very top
st.image(BASE_DIR / "assets" / "hollyowl.jpg", use_container_width=True)


st.title("Music Marketing Analytics - Correlating paid advertising with streaming performance across platforms")

# ============================================
# INTRO SECTION
# ============================================

st.markdown("""
This was quite the exciting project: new, both fun and challenging. I had two datasets at my disposal: a marketing file with campaigns running from January to August 2023, and a streaming file with Spotify data from January to March 2023. The marketing metrics were attributable to specific campaigns; the streaming ones were not — these are simply all streams for a given artist on a given day. I wanted to share as much as possible from this case study while being discreet with the data, so I pseudonymized the whole thing using deterministic hashing and David Lynch-related names.

This started as a tribute to the beloved director who passed away last year. Music was such a vehicle for mood-setting in Lynch's work, where people often play detectives (sometimes professional ones), so it felt right to feature songs by him and sublime collaborators Angelo Badalamenti and Julee Cruise.

But being absolutely new to marketing and rather off-apps, I found myself in a maze of mysteries and things that weren't what they seemed at first. Hence, the superimposed images you see in this dashboard are as much a tribute to David Lynch as they are an attestation of my trying to find my way out of the analytical woods — and a visual rendering of how different campaigns' overlapping influence affects streaming numbers and patterns.
""")

# 2. good_witch_laura_sycamore.jpg - after opening paragraph
st.image(BASE_DIR / "assets" / "good_witch_laura_sycamore.jpg", use_container_width=True)


# Spotify playlist embed with shuffle
st.components.v1.html(
    """
    <iframe style="border-radius:12px" 
            src="https://open.spotify.com/embed/playlist/6oIVta944ygc5UuSTVAfqa?utm_source=generator&theme=0" 
            width="100%" 
            height="152" 
            frameBorder="0" 
            allowfullscreen="" 
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
            loading="lazy">
    </iframe>
    """,
    height=160
)
st.markdown("### 🔍 Data Notes")
st.markdown("""
**Pseudonymization.** With ~1,000 songs to rename, I prompted Claude to take on what would have been a fun dorky task in smaller doses. The result: we get some unfortunate names like "Abstract_Art" but miss staples like "Invitation_to_Love."! But I'm being ungrateful. I would never have done this mammoth task without Claude, let alone the rest of the coding :D 

**Similar song names.** You'll see titles like "We_Dont_Stop_Here" and "We_Dont_Stop_Here_She_Said." These are <u>NOT</u> iterations of the same song. I kept them because catalogs in real life often have songs with identical or near-identical titles.

**Canonization.** Canonization in fact happened early (in notebook 01 with the truly original files) so I'd have stable names for pseudonymization (notebook 02). A side effect: products with non-canonical names (e.g., "Umbrella live feat. Jay-Z" instead of "Umbrella") became blanks after hashing. That's why you'll see 4,866 blanks in the pseudonymized marketing file. But just scroll right — the canonical name is there.

**Streaming perturbation.** There's a ±10% random perturbation on streaming numbers for additional data protection. So: names are fake, streaming numbers are slightly off, marketing metrics are real.
""", unsafe_allow_html=True)

# ============================================
# NAVIGATION
# ============================================

tab1, tab2 = st.tabs(["📊 Platform Performance", "📈 Streaming Explorer"])

# ============================================
# TAB 1: PLATFORM PERFORMANCE
# ============================================

with tab1:
    st.header("If you have budget to drive listeners to a new release, which platform delivers the most value — and does it depend on what you're promoting?")
    
    st.markdown("""
This view compares platform performance for **traffic campaigns** specifically. We break it down by product type (track, album, playlist, URL) because platforms perform differently depending on the destination.
**Why traffic campaigns only.** Traffic was the only objective with enough data across platforms and product types to make meaningful comparisons. Awareness and engagement campaigns had too many gaps or low-confidence pairs.
    """)
    
    # Questions for paid media managers
    st.markdown("### 🌹 Questions for Paid Media Managers")
    st.markdown("""
- Do you agree with this analytical grain — platform × objective × product type — for evaluating performance? My idea was to make findings actionable: what is being promoted, with which goal, on which platform.

- Other choices I made:

- Thresholds: Was I too conservative with confidence tiers? Would you trust smaller samples?
- Awareness vs. reach: I kept these separate even though some platforms use only one term (TikTok says "reach" exclusively). My read is they optimize for subtly different things — impressions/exposure (awareness) vs. unique exposure/new audiences (reach). But I'd value a practitioner's take!
    """)
    
    # Methodology notes - BEFORE the chart
    st.markdown("### ☕ Methodology notes")
    st.markdown("""
**Metrics shown:**
- **Traffic per day:** Total clicks ÷ campaign duration. Normalizes for different campaign lengths.
- **Cost per click:** Spend ÷ clicks. Lower is better.
- **Click-through rate (CTR):** Clicks ÷ impressions. Measures how compelling the ad was.

**Bar color saturation = confidence tiers:**
- **Solid colors (A_reliable):** 10+ campaigns or 150+ total campaign-days. Patterns here are meaningful.
- **Faded (B_indicative):** 6–9 campaigns or 90–149 days. Directional signal, not definitive.
- **Very faded (C_exploratory):** Fewer than 6 campaigns. Interesting but anecdotal — don't bet budget on it.

**How composite metrics were calculated:**

Different platforms report metrics differently, so I built standardized composites (this was also part of the mystery ride, they reflect my reading of the numbers):
- **Awareness:** The greater of impressions or reach (to avoid double-counting), plus video views
- **Traffic:** Link clicks + conversions
- **Engagement:** Platform-specific — Meta reports "interactions" as a rollup; TikTok breaks out comments, likes, shares, follows; Google/YouTube interactions include video views, so I subtracted those; Snapchat doesn't report engagement metrics

**The reach aggregation problem:** Within a single campaign, reach is deduplicated. But when aggregating across campaigns, there's no clean solution — summing overcounts (audiences overlap), taking max undercounts (ignores smaller campaigns). I used max as a conservative lower bound. Worth bearing in mind when interpreting.

**Platform overlap caveat:** Many campaigns run concurrently across platforms. When Instagram appears to outperform Snapchat, some of that lift might be other platforms doing invisible work. The underlying data flags high-overlap campaigns, but this view aggregates across them. Interpret directionally.

**What's excluded by design:**
- *Content type:* ~40% is categorized as "Other," limiting interpretability
- *Audience type:* Applied inconsistently across objectives — it appears subordinate to campaign goal rather than a defining strategic variable
    """)
    
    # 3. fire_tulips.jpg - after Methodology notes, before charts
    st.image(BASE_DIR / "assets" / "fire_tulips.jpg", use_container_width=True)
    
    # Filter to traffic only
    df_traffic = platform_summary[platform_summary['objective'] == 'traffic'].copy()
    
    # Create display labels - just abbreviation, details in hover
    df_traffic['platform_short'] = df_traffic['platform_category'].map(platform_abbrev)
    df_traffic['bar_label'] = df_traffic['platform_short']
    df_traffic['hover_info'] = df_traffic.apply(
        lambda row: f"{row['platform_short']} ({int(row['campaign_count'])} campaigns, {int(row['total_campaign_duration_days'])} days)", 
        axis=1
    )
    
    product_types = ['Track URI', 'Album URI', 'Playlist URI', 'URL']
    
    # All 3 metrics shown vertically
    metrics_config = [
        ('traffic_per_day', 'Traffic Volume (clicks per day)', True),
        ('cost_per_click', 'Cost Efficiency (€ per click)', False),
        ('click_through_rate_pct', 'Click-Through Rate (%)', True)
    ]
    
    for metric_col, metric_title, higher_is_better in metrics_config:
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=product_types,
            horizontal_spacing=0.08
        )
        
        for i, product in enumerate(product_types, 1):
            df_prod = df_traffic[
                (df_traffic['product_number_type'] == product) & 
                (df_traffic[metric_col].notna()) &
                (df_traffic[metric_col] > 0)
            ].sort_values(metric_col, ascending=not higher_is_better)
            
            if len(df_prod) == 0:
                continue
            
            colors = [
                platform_colors_base.get(p, '#888888') 
                for p in df_prod['platform_category']
            ]
            opacities = [
                confidence_alpha.get(c, 0.5) 
                for c in df_prod['confidence_tier']
            ]
            
            fig.add_trace(
                go.Bar(
                    y=df_prod['bar_label'],
                    x=df_prod[metric_col],
                    orientation='h',
                    marker=dict(
                        color=colors,
                        opacity=opacities
                    ),
                    customdata=df_prod['hover_info'],
                    hovertemplate=(
                        "<b>%{customdata}</b><br>" +
                        f"{metric_title}: " + "%{x:.2f}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            height=280,
            showlegend=False,
            title_text=metric_title,
            title_x=0.5,
            margin=dict(t=50, b=20, l=10, r=10)
        )
        
        for i in range(1, 5):
            fig.update_yaxes(title_text="", row=1, col=i)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Findings and Recommendations
    st.markdown("---")
    
    st.markdown("### Findings & Recommendations")
    
    st.markdown("""
**Track:** TikTok delivers impressive volume at the lowest cost, but its click-through rate is the weakest. Facebook, despite fewer campaigns, shows the highest CTR with solid volume (compared to Instagram, not volume-king TikTok) and slightly better cost efficiency than Instagram. **Consider shifting some TikTok budget to Facebook to balance reach with intent.**

**Album:** Instagram and Facebook are pulling their weight here - good volume, reasonable costs. Snapchat is the outlier: 110 days of campaigns yielding the lowest volume, highest cost per click, and mediocre CTR. **Snapchat's budget would work harder on Facebook.**

**Playlist:** Instagram leads on volume with decent efficiency. Snapchat again underperforms - high cost, low volume, and while it has the best CTR, that doesn't compensate. TikTok shows promise on cost and volume despite low CTR; worth testing with a small budget increase. **Reduce Snapchat, maintain Instagram, experiment with TikTok.**

**URL:** Facebook shows an interesting signal: lowest cost per click and highest CTR, but limited data (39 days). TikTok has volume but very low engagement. **Worth expanding Facebook testing to see if these early results hold.**

*These findings reflect Q1 campaign data and don't account for audience demographics or content type, which may justify platform choices not captured here.*
    """)
    
    # 4. blue_sandy_jeffrey.jpg - after Findings and Recommendations
    st.image(BASE_DIR / "assets" / "blue_box_new_couple_sandy_jeffrey.jpg", use_container_width=True)

# ============================================
# TAB 2: STREAMING EXPLORER
# ============================================

with tab2:
    # 5. lost_jitterbug.jpg - before Product Manager Dashboard
    st.image(BASE_DIR / "assets" / "lost_jitterbug.jpg", use_container_width=True)
    
    st.header("When campaigns run, what happens to streaming? And where do those streams come from?")
    
    st.markdown("""
This interactive chart was designed for product managers who want to understand how marketing activity correlates with streaming patterns. Select an artist to see their daily streams over time, with campaign periods overlaid. The stacked area shows *where* streams originate — search, radio, playlists, collections, and so on.

**Why correlation, not attribution.** Our streaming data doesn't isolate ad-exposed listeners, so we can't say who became a listener because of an ad versus who was already a fan. On top of that, campaigns for the same artist often run simultaneously across platforms, making it harder to credit any single effort. What we *can* do is observe patterns: do streams spike when campaigns start? Do certain sources grow more than others? Does activity persist after campaigns end, or does it mitigate what would otherwise be a steepening decline?
    """)
    
    # Methodology notes - BEFORE the chart
    st.markdown("### 🍩 Methodology notes")
    st.markdown("""
**What streams are shown?** All streams for the selected artist, across their entire catalog. We don't filter to just the promoted product — partly because all streams generate revenue, and partly because campaigns often lift an artist's whole catalog (halo effect). You can use the song filter to isolate specific tracks if you want to check whether the promoted product itself is getting traction.

**Why this matters for non-track campaigns:** Album, playlist, and URL campaigns point to containers or external links, not individual tracks. There's no clean way to attribute track-level streams to these. The artist-level view sidesteps this problem.

**A caveat:** This means you might see strong streams during a campaign period even if the promoted song specifically isn't doing much. The per-song breakdown (visible when you filter) helps you check whether the advertised product is carrying its weight or whether it's old catalog doing the work.

**What to look for:**
- Streams rising during campaign periods (colored bands)
- Shifts in source mix — e.g., search increasing might suggest discovery behavior
- Whether gains hold in the decay period or drop back to baseline
    """)
    
    # 6. tears_of_joy_waterfall.jpg - before Select Artist
    st.image(BASE_DIR / "assets" / "tears_of_joy_waterfall.jpg", use_container_width=True)
    
    # Get artists with both campaign and streaming data
    campaign_artists = set(campaign_summary['canonical_artist'].unique())
    stream_artists = set(daily_streams['canonical_artist'].unique())
    available_artists = sorted(campaign_artists & stream_artists)
    
    # Default to Catherine_Martell (interesting example with multiple campaigns)
    default_artist = 'Catherine_Martell'
    default_index = available_artists.index(default_artist) if default_artist in available_artists else 0
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_artist = st.selectbox(
            "Select Artist",
            available_artists,
            index=default_index
        )
    
    # Filter data for selected artist
    artist_campaigns = campaign_summary[campaign_summary['canonical_artist'] == selected_artist].copy()
    artist_streams = daily_streams[daily_streams['canonical_artist'] == selected_artist].copy()
    
    # Get only product types that this artist actually has campaigns for
    artist_product_types = sorted(artist_campaigns['product_number_type'].dropna().unique().tolist())
    product_type_options = ['All'] + artist_product_types
    
    with col2:
        product_type_filter = st.selectbox(
            "Advertised Product Type",
            product_type_options
        )
    
    with col3:
        show_by_source = st.checkbox("Show stream sources", value=True)
    
    # Apply product type filter
    if product_type_filter != 'All':
        artist_campaigns = artist_campaigns[artist_campaigns['product_number_type'] == product_type_filter]
    
    # Song filter
    available_songs = sorted(artist_streams['canonical_product'].unique())
    selected_songs = st.multiselect(
        "Filter to specific songs (optional)",
        available_songs,
        default=[]
    )
    
    if selected_songs:
        artist_streams_filtered = artist_streams[artist_streams['canonical_product'].isin(selected_songs)]
    else:
        artist_streams_filtered = artist_streams
    
    # Prepare streams
    if show_by_source:
        streams_daily = (
            artist_streams_filtered
            .groupby(['report_date', 'source_name'])['streams']
            .sum()
            .reset_index()
        )
        if not streams_daily.empty:
            pivot_streams = streams_daily.pivot(index='report_date', columns='source_name', values='streams').fillna(0)
        else:
            pivot_streams = pd.DataFrame()
    else:
        streams_daily = (
            artist_streams_filtered
            .groupby('report_date')['streams']
            .sum()
            .reset_index()
        )
        pivot_streams = pd.DataFrame()
    
    # Build figure
    fig = go.Figure()
    
    # Add streams
    if show_by_source and not pivot_streams.empty:
        daily_totals = pivot_streams.sum(axis=1)
        
        for source in pivot_streams.columns:
            source_data = pivot_streams[source]
            percentages = [
                (float(source_data[d]) / float(daily_totals[d]) * 100.0) if daily_totals[d] > 0 else 0.0
                for d in source_data.index
            ]
            
            color = source_colors.get(source.lower(), '#CCCCCC')
            
            fig.add_trace(go.Scatter(
                x=pivot_streams.index,
                y=source_data,
                name=source.capitalize(),
                mode='lines',
                stackgroup='one',
                line=dict(width=0),
                fillcolor=color,
                hovertemplate=(
                    f'<b>{source}</b><br>'
                    f'Streams: %{{y:,.0f}}<br>'
                    f'Share: %{{customdata:.1f}}%'
                    '<extra></extra>'
                ),
                customdata=percentages
            ))
    elif not streams_daily.empty:
        fig.add_trace(go.Scatter(
            x=streams_daily['report_date'],
            y=streams_daily['streams'],
            mode='lines',
            name='Total Streams',
            line=dict(color='#1DB954', width=2),
            fill='tozeroy'
        ))
    
    # Add campaign periods
    for idx, campaign in artist_campaigns.iterrows():
        platform = campaign['platform_category']
        color = platform_colors_bg.get(platform, 'rgba(100, 100, 100, 0.25)')
        
        if pd.notna(campaign['campaign_start_date']) and pd.notna(campaign['campaign_end_date']):
            fig.add_vrect(
                x0=campaign['campaign_start_date'],
                x1=campaign['campaign_end_date'],
                fillcolor=color,
                layer='below',
                line_width=0,
                annotation_text=platform_abbrev.get(platform, platform[:2].upper()),
                annotation_position='top left',
                annotation_font_size=10
            )
    
    # Add campaign start markers - GROUP overlapping campaigns
    if show_by_source and not pivot_streams.empty:
        max_streams = pivot_streams.sum(axis=1).max()
    elif not streams_daily.empty:
        max_streams = streams_daily['streams'].max()
    else:
        max_streams = 1000
    
    base_marker_y = max_streams * 1.05
    marker_spacing = max_streams * 0.08
    
    # Group campaigns by start date
    campaigns_by_date = defaultdict(list)
    for idx, campaign in artist_campaigns.iterrows():
        if pd.notna(campaign['campaign_start_date']):
            campaigns_by_date[campaign['campaign_start_date']].append(campaign)
    
    campaign_x = []
    campaign_y = []
    campaign_hovers = []
    campaign_colors_list = []
    
    for start_date, campaigns_list in campaigns_by_date.items():
        for i, campaign in enumerate(campaigns_list):
            campaign_x.append(start_date)
            campaign_y.append(base_marker_y + (i * marker_spacing))
            
            duration = (campaign['campaign_end_date'] - campaign['campaign_start_date']).days + 1
            spend_per_day = campaign['spend'] / duration if duration > 0 else 0
            
            # Calculate per-day metrics
            traffic_per_day = campaign.get('total_traffic_metrics', 0) / duration if duration > 0 else 0
            awareness_per_day = campaign.get('total_awareness_metrics', 0) / duration if duration > 0 else 0
            engagement_per_day = campaign.get('total_engagement_metrics', 0) / duration if duration > 0 else 0
            
            hover_text = (
                f"<b>{campaign['platform_category'].upper()}</b><br>"
                f"Objective: {campaign['objective']}<br>"
                f"Product Type: {campaign.get('product_number_type', 'N/A')}<br>"
                f"Product: {campaign.get('canonical_product', 'N/A')}<br>"
                f"Start: {campaign['campaign_start_date'].strftime('%Y-%m-%d')}<br>"
                f"Duration: {duration} days<br>"
                f"Spend/day: €{spend_per_day:,.2f}<br>"
                f"Traffic/day: {traffic_per_day:,.0f}<br>"
                f"Awareness/day: {awareness_per_day:,.0f}<br>"
                f"Engagement/day: {engagement_per_day:,.0f}"
            )
            campaign_hovers.append(hover_text)
            campaign_colors_list.append(platform_colors_base.get(campaign['platform_category'], '#888888'))
    
    if campaign_x:
        fig.add_trace(go.Scatter(
            x=campaign_x,
            y=campaign_y,
            mode='markers',
            marker=dict(
                symbol='diamond',
                size=12,
                color=campaign_colors_list,
                line=dict(color='white', width=2)
            ),
            name='Campaign Start',
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=campaign_hovers,
            showlegend=True
        ))
    
    # Stats
    products_advertised = artist_campaigns['canonical_product'].nunique()
    products_streamed = artist_streams_filtered['canonical_product'].nunique()
    total_campaigns = len(artist_campaigns)
    
    fig.update_layout(
        title=f"{selected_artist}: {total_campaigns} campaigns across {products_advertised} products | {products_streamed} products streamed",
        xaxis_title="Date",
        yaxis_title="Daily Streams",
        hovermode='x unified',
        height=550,
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        margin=dict(t=50, b=100, l=60, r=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hint about source selection
    st.caption("💡 Click on stream sources in the legend to show/hide them for closer inspection.")
    
    # Campaign summary table with per-day metrics
    if not artist_campaigns.empty:
        st.markdown("**Campaign Details:**")
        campaign_display = artist_campaigns.copy()
        campaign_display['Duration'] = campaign_display['campaign_duration_days']
        campaign_display['Spend/day'] = (campaign_display['spend'] / campaign_display['campaign_duration_days']).round(2)
        campaign_display['Traffic/day'] = (campaign_display['total_traffic_metrics'] / campaign_display['campaign_duration_days']).round(0)
        campaign_display['Awareness/day'] = (campaign_display['total_awareness_metrics'] / campaign_display['campaign_duration_days']).round(0)
        campaign_display['Engagement/day'] = (campaign_display['total_engagement_metrics'] / campaign_display['campaign_duration_days']).round(0)
        
        campaign_display = campaign_display[['platform_category', 'objective', 'product_number_type', 
                                              'canonical_product', 'campaign_start_date', 'Duration',
                                              'Spend/day', 'Traffic/day', 'Awareness/day', 'Engagement/day']].copy()
        campaign_display.columns = ['Platform', 'Objective', 'Product Type', 'Product', 'Start', 'Days',
                                    'Spend/day (€)', 'Traffic/day', 'Awareness/day', 'Engagement/day']
        campaign_display['Start'] = campaign_display['Start'].dt.strftime('%Y-%m-%d')
        st.dataframe(campaign_display, use_container_width=True, height=200, hide_index=True)
    
    # Platform legend
    st.markdown("**Platform colors:** " + " · ".join([
        f"<span style='color:{color}'>{platform_abbrev[p]}</span>" 
        for p, color in platform_colors_base.items()
    ]), unsafe_allow_html=True)
    
    # Summary stats
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_streams = artist_streams_filtered['streams'].sum()
    total_spend = artist_campaigns['spend'].sum()
    total_ad_days = artist_campaigns['campaign_duration_days'].sum() if 'campaign_duration_days' in artist_campaigns.columns else 0
    
    with col1:
        st.metric("Total Artist Streams", f"{total_streams:,.0f}")
    with col2:
        st.metric("Total Campaigns", total_campaigns)
    with col3:
        st.metric("Total Advertising Days", f"{total_ad_days:,.0f}")
    with col4:
        st.metric("Total Spend", f"€{total_spend:,.0f}")
    
    # Song breakdown table
    st.markdown("---")
    
    
    st.subheader("Song-level Stream Breakdown")
    st.caption("See which songs are driving streams — is it the promoted product or catalog favorites?")
    
    song_summary = (
        artist_streams
        .groupby('canonical_product')['streams']
        .sum()
        .reset_index()
        .rename(columns={'canonical_product': 'Song', 'streams': 'Total Streams'})
        .sort_values('Total Streams', ascending=False)
    )
    
    # Mark which songs were advertised
    advertised_products = set(artist_campaigns['canonical_product'].dropna().unique())
    song_summary['Advertised'] = song_summary['Song'].apply(lambda x: '✓' if x in advertised_products else '')
    
    # Add percentage
    total = song_summary['Total Streams'].sum()
    song_summary['% of Total'] = (song_summary['Total Streams'] / total * 100).round(1)
    
    # Reorder columns
    song_summary = song_summary[['Song', 'Advertised', 'Total Streams', '% of Total']]
    
    st.dataframe(
        song_summary,
        use_container_width=True,
        height=300,
        hide_index=True
    )
    
    # Asking Product Managers 
    st.markdown("### 🚬 Questions for Product Managers")
    st.markdown("""
- What patterns would you want to detect that this view doesn't currently surface?
- What's missing here that would help you make a go/no-go decision on a campaign strategy?
- Is the artist-level view useful, or would you prefer only product-level streaming data? Currently you can filter for product-level streaming but the default is artist-level view, with the assumption that campaigns influence people to listen to other (current or past) songs of a given artist.
    """)
    
    # 7. final_curtain.jpg - at the end after Song-level Stream Breakdown
    st.image(BASE_DIR / "assets" / "final_curtain.jpg", use_container_width=True)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85em;'>
    Music Marketing Analytics Dashboard · Q1 2023 Campaign Data · 
    Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
