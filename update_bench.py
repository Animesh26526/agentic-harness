with open("app/pages/2_Benchmark.py", "r") as f:
    code = f.read()

target = """        st.write("")
        # Charts for User Results
        uc_col1, uc_col2 = st.columns(2)"""

new_code = """        st.write("")
        from harness.analytics import get_memory_stats
        mem_stats = get_memory_stats(user_on, db_path=db_path)
        
        with st.expander("🧠 Harness Memory Performance", expanded=True):
            st.markdown(f'''
            | Memory Metric | Live Value |
            | :--- | :---: |
            | **Memory Assisted Retries** | `{mem_stats['memory_assisted_retries']}` |
            | **Memory Usage Rate** | `{mem_stats['memory_usage_rate'] * 100:.1f}%` |
            | **Memory Assisted Success Rate** | `{mem_stats['memory_assisted_success_rate'] * 100:.1f}%` |
            | **Avg. Reliability Improvement** | `+{mem_stats['avg_reliability_improvement']:.3f}` |
            ''')
            
            if mem_stats['top_learned_strategies']:
                st.markdown("**Top Learned Strategies (Global):**")
                for s in mem_stats['top_learned_strategies']:
                    st.markdown(f"- {s['strategy']} (Success Rate: {int(s['success_rate']*100)}%)")
            else:
                st.markdown("*No strategies learned yet. Run more benchmarks to accumulate memory.*")

        st.write("")
        # Charts for User Results
        uc_col1, uc_col2 = st.columns(2)"""

code = code.replace(target, new_code)

with open("app/pages/2_Benchmark.py", "w") as f:
    f.write(code)
