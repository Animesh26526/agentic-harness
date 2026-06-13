import re

with open('app/pages/2_Benchmark.py', 'r') as f:
    content = f.read()

# Modify get_latest_completed_benchmark_pair to enforce total_samples = 40
content = content.replace(
    "AND total_samples > 0",
    "AND total_samples = 40"
)

# Extract the body of "Your Experiment Results" and use it to replace "Official Benchmark Results"
# First we find the start of Official Benchmark Results
start_idx = content.find("# 1. Official Benchmark Results (Static Showcase Baseline Evidence)")
end_idx = content.find("# 3. Live Benchmark Execution Controls")

if start_idx != -1 and end_idx != -1:
    new_section = """# -----------------------------------------------------------------------------
# 1. Official Benchmark Results (Live 40-Sample Showcase)
# -----------------------------------------------------------------------------
st.write("")
st.markdown("## 🏆 Official Benchmark Results")
st.write(
    "These are the baseline project evidence metrics generated across the official "
    "**40-sample benchmark dataset** (10 samples per category). This represents the verified "
    "framework capacity."
)

user_off, user_on = get_latest_completed_benchmark_pair(db_path)

if user_off and user_on:
    try:
        # Load live database stats
        usr_sr_off = get_success_rate(user_off, db_path=db_path)
        usr_sr_on = get_success_rate(user_on, db_path=db_path)
        usr_rel_off = get_average_reliability(user_off, db_path=db_path)
        usr_rel_on = get_average_reliability(user_on, db_path=db_path)
        usr_err_reduction = get_error_reduction_rate(user_off, user_on, db_path=db_path)
        usr_recovery = get_recovery_rate(user_off, user_on, db_path=db_path)
        
        # Load category breakdown
        cat_list = ["structured_json", "constraint_following", "factual_qa", "extraction_math"]
        breakdown_off = get_category_breakdown(user_off, db_path=db_path)
        breakdown_on = get_category_breakdown(user_on, db_path=db_path)
        usr_cat_off_rel = [breakdown_off.get(cat, {}).get("avg_reliability", 0.0) for cat in cat_list]
        usr_cat_on_rel = [breakdown_on.get(cat, {}).get("avg_reliability", 0.0) for cat in cat_list]
        
        # Load retry distribution
        retry_dist = get_retry_distribution(user_on, db_path=db_path)
        usr_retry_counts = [
            retry_dist.get(0, 0),
            retry_dist.get(1, 0),
            retry_dist.get(2, 0),
            retry_dist.get(3, 0)
        ]
        
        # Determine total sample count
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM run_logs WHERE run_id = ?", (user_on,))
        usr_total_samples = cursor.fetchone()[0]
        cursor.execute("SELECT overall_reliability FROM run_logs WHERE run_id = ?", (user_on,))
        usr_raw_scores_on = [float(row[0]) for row in cursor.fetchall()]
        conn.close()

        st.markdown(f'<div class="status-banner banner-live">🛡️ Active SQLite metrics connected! Displaying official experiment data from runs: <code>OFF ({user_off})</code> vs <code>ON ({user_on})</code>.</div>', unsafe_allow_html=True)
        
        # Metric summary grid for User Results
        um_col1, um_col2, um_col3, um_col4, um_col5 = st.columns(5)
        
        with um_col1:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-title">Success Rate (OFF)</div>
                <div class="metric-value">{usr_sr_off * 100:.1f}%</div>
                <div class="metric-sub">Raw Baseline</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with um_col2:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-title">Success Rate (ON)</div>
                <div class="metric-value" style="color: #10B981;">{usr_sr_on * 100:.1f}%</div>
                <div class="metric-sub">Harness Enabled</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with um_col3:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-title">Reliability Delta</div>
                <div class="metric-value" style="color: #00F2FE;">+{usr_rel_on - usr_rel_off:+.3f}</div>
                <div class="metric-sub">OFF: {usr_rel_off:.3f} | ON: {usr_rel_on:.3f}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with um_col4:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-title">Error Reduction</div>
                <div class="metric-value">{usr_err_reduction * 100:.1f}%</div>
                <div class="metric-sub">Failures Corrected</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with um_col5:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-title">Recovery Rate</div>
                <div class="metric-value">{usr_recovery * 100:.1f}%</div>
                <div class="metric-sub">Self-Correction Success</div>
            </div>
            ''', unsafe_allow_html=True)
            
        # Dynamically calculate effectiveness metrics
        from harness.analytics import get_harness_effectiveness
        eff = get_harness_effectiveness(user_on, db_path=db_path)
        
        st.write("")
        with st.expander("🛡️ Live Harness Effectiveness & Correctness Audit", expanded=True):
            st.markdown(f'''
            This audit panel dynamically computes key precision and intervention accuracy metrics for the connected official run (`{user_on}`).
            
            | Effectiveness Metric | Live Value | Description |
            | :--- | :---: | :--- |
            | **Total Runs** | `{eff['total_runs']}` | Total number of test samples executed in this live experiment. |
            | **Raw Pass Rate** | `{eff['raw_pass_rate'] * 100:.1f}%` | Success rate of the raw model output on its first attempt. |
            | **Harness Pass Rate** | `{eff['harness_pass_rate'] * 100:.1f}%` | Final success rate after self-correcting interventions. |
            | **Retry Rate** | `{eff['retry_rate'] * 100:.1f}%` | Percentage of runs that triggered a self-correcting retry. |
            | **Retry Success Rate** | `{eff['retry_success_rate'] * 100:.1f}%` | Percentage of retried runs that successfully recovered. |
            | **False Retry Rate** | `{eff['false_retry_rate'] * 100:.1f}%` | Interventions on already acceptable responses (harness noise). |
            | **False Pass Rate** | `{eff['false_pass_rate'] * 100:.1f}%` | Responses accepted with outstanding violations (harness misses). |
            ''')
            
        # Charts for User Results
        uc_col1, uc_col2 = st.columns(2)
        
        with uc_col1:
            st.markdown("#### Category Improvement Breakdown")
            fig_cat_usr = go.Figure()
            fig_cat_usr.add_trace(go.Bar(
                x=cat_list,
                y=usr_cat_off_rel,
                name="Harness OFF",
                marker_color="#EF4444"
            ))
            fig_cat_usr.add_trace(go.Bar(
                x=cat_list,
                y=usr_cat_on_rel,
                name="Harness ON",
                marker_color="#00F2FE"
            ))
            fig_cat_usr.update_layout(
                barmode='group',
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Average Reliability Score",
                yaxis=dict(range=[0.0, 1.05]),
                margin=dict(l=40, r=40, t=10, b=40)
            )
            st.plotly_chart(fig_cat_usr, use_container_width=True)
            
        with uc_col2:
            st.markdown("#### Correction Retry Distribution")
            fig_dist_usr = go.Figure()
            fig_dist_usr.add_trace(go.Bar(
                x=["Attempt 1", "Attempt 2", "Attempt 3", "Failed"],
                y=usr_retry_counts,
                marker_color=["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]
            ))
            fig_dist_usr.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Query Count",
                xaxis_title="Resolution Step",
                margin=dict(l=40, r=40, t=10, b=40)
            )
            st.plotly_chart(fig_dist_usr, use_container_width=True)
            
        # Histogram for User Results
        st.write("")
        st.markdown("#### Harness ON Overall Reliability Distribution")
        fig_hist_usr = px.histogram(
            x=usr_raw_scores_on,
            nbins=10,
            range_x=[0.0, 1.0],
            color_discrete_sequence=["#00F2FE"]
        )
        fig_hist_usr.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Reliability Score",
            yaxis_title="Query Count",
            bargap=0.08,
            margin=dict(l=40, r=40, t=10, b=40)
        )
        st.plotly_chart(fig_hist_usr, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading official experiment results: {str(e)}")
else:
    st.info("ℹ️ No official 40-sample benchmark results found in the database. Run the official dataset to populate.")

# -----------------------------------------------------------------------------
"""
    content = content[:start_idx] + new_section + content[end_idx:]
    with open('app/pages/2_Benchmark.py', 'w') as f:
        f.write(content)
