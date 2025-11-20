#!/usr/bin/env python3
"""
Streamlit NER Model Comparison Demo
Side-by-side comparison of Base Qwen3-4B vs LoRA Zero3 Fine-tuned model
"""

import streamlit as st
import pandas as pd
import json
import time
import sys
import os
from pathlib import Path

# Add the demo directory to Python path
demo_dir = Path(__file__).parent
sys.path.insert(0, str(demo_dir))

from model_comparison import NERComparisonClient, ModelResult
from visualization import (
    create_inference_speed_chart,
    create_entity_type_distribution,
    create_comparison_radar_chart,
    create_entity_comparison_table
)

# Page configuration
st.set_page_config(
    page_title="SFT-ner Model Comparison Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .model-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .base-model-card {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .lora-model-card {
        background-color: #fce4ec;
        border-left: 5px solid #e91e63;
    }
    .entity-highlight {
        background-color: #ffff00;
        font-weight: bold;
        padding: 2px 4px;
        border-radius: 3px;
        color: #000000 !important;  /* Dark color for better contrast */
    }
    .metric-box {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .improvement-badge {
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.9em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'client' not in st.session_state:
    import os
    base_api_url = os.getenv('BASE_API_URL', 'http://localhost:8003')
    lora_api_url = os.getenv('LORA_API_URL', 'http://localhost:8002')
    st.session_state.client = NERComparisonClient(
        base_api_url=base_api_url,
        lora_api_url=lora_api_url
    )

if 'test_results' not in st.session_state:
    st.session_state.test_results = None

if 'batch_mode' not in st.session_state:
    st.session_state.batch_mode = False

# Load test cases
@st.cache_data
def load_test_cases():
    """Load test cases from JSON file"""
    test_cases_path = demo_dir.parent / "examples" / "test_cases.json"
    if test_cases_path.exists():
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# Main header
st.markdown("<h1 class='main-header'>🎯 SFT-ner Model Comparison Demo</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; font-size: 1.2em; color: #666; margin-bottom: 30px;">
    <span style="font-size: 0.9em;">对比基底模型 vs LoRA Zero3 Fine-tuned 适配器</span>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("控制面板")

    st.markdown("---")

    # Mode selection
    st.subheader("🎛️ 运行模式")

    mode = st.radio(
        "选择模式",
        ["单文本测试", "批量测试 (10个用例)"],
        index=0
    )

    st.session_state.batch_mode = mode == "批量测试 (10个用例)"

    st.markdown("---")

    # Input method
    st.subheader("📝 输入方式")

    if not st.session_state.batch_mode:
        input_method = st.selectbox(
            "选择输入方式",
            ["自定义输入", "样例输入", "预设测试用例"]
        )

        # Initialize session state for sample text if not exists
        if 'sample_text' not in st.session_state:
            st.session_state.sample_text = ""

        if input_method == "预设测试用例":
            test_cases = load_test_cases()
            if test_cases:
                case_options = [
                    f"用例 {i+1}: {case['input'][:50]}..."
                    for i, case in enumerate(test_cases)
                ]
                selected_case = st.selectbox("选择测试用例", range(len(test_cases)),
                                           format_func=lambda x: case_options[x])
                input_text = test_cases[selected_case]['input']
            else:
                st.error("未找到测试用例文件")
                input_text = ""
        elif input_method == "样例输入":
            import json
            import random

            # Load test samples from the JSON file
            test_file_path = "/home/ubuntu/SFT-ner/military-ner-project/data/test_processed.json"
            if os.path.exists(test_file_path):
                with open(test_file_path, 'r', encoding='utf-8') as f:
                    all_cases = json.load(f)

                # Randomly select 5-6 samples
                sample_size = min(6, len(all_cases))
                random_samples = random.sample(all_cases, sample_size)

                st.markdown("**点击以下样例快速填充：**")

                # Create buttons for each sample
                for i, sample in enumerate(random_samples, 1):
                    sample_text_preview = sample['input'][:100] + "..." if len(sample['input']) > 100 else sample['input']

                    if st.button(f"样例 {i}: {sample_text_preview}", key=f"sample_{i}", use_container_width=True):
                        st.session_state.sample_text = sample['input']
                        st.rerun()

                # Display the text area with the selected sample
                input_text = st.text_area(
                    "输入军事文本 (已填充样例)",
                    value=st.session_state.sample_text,
                    height=200
                )
            else:
                st.error("未找到样例文件")
                input_text = ""
        else:  # 自定义输入
            input_text = st.text_area(
                "输入军事文本",
                height=200,
                placeholder="例如: 美国(-39.01,-141.10)拥有448枚准备发射的洲际弹道导弹..."
            )
    else:
        st.info("批量测试将运行所有10个预设测试用例")
        input_text = ""

    st.markdown("---")

    # Action buttons
    st.subheader("⚡ 操作")

    if st.button("🚀 开始分析", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    else:
        st.session_state.run_analysis = False

    if st.button("🔄 清空结果", use_container_width=True):
        st.session_state.test_results = None
        st.rerun()

    st.markdown("---")

    # Information
    st.subheader("ℹ️ 关于项目")

    st.info("""
    **项目名称**: SFT-ner (Supervised Fine-Tuning for Named Entity Recognition)

    **基底模型**: Qwen3-4B (8.0B参数)

    **LoRA适配器**: ner_zero3 (ZeRO3优化)

    **实体类型**:
    - 🎯 军事装备 (Weapon/Equipment)
    - 🗺️ 地理位置 (Location with Coordinates)
    - 🏢 组织名称 (Organization)
    - 👤 人名 (Person)

    **技术**: LoRA + ZeRO3 + vLLM
    """)

# Main area
if st.session_state.batch_mode:
    st.subheader("📊 批量测试模式")
    st.info("批量测试将运行所有10个测试用例，并生成详细的性能对比报告")

    if st.session_state.run_analysis:
        with st.spinner("正在运行批量测试，请稍候..."):
            test_cases = load_test_cases()

            if not test_cases:
                st.error("未找到测试用例")
            else:
                batch_results = []
                progress_bar = st.progress(0)

                for idx, case in enumerate(test_cases):
                    progress = (idx + 1) / len(test_cases)
                    progress_bar.progress(progress)

                    text = case['input']

                    with st.expander(f"测试用例 {idx + 1}", expanded=False):
                        st.markdown(f"**输入文本:** {text[:200]}...")

                        # Run comparison
                        base_result, lora_result = st.session_state.client.extract_entities_both(text)

                        # Calculate comparison
                        comparison = st.session_state.client.compare_entities(
                            base_result.entities, lora_result.entities
                        )

                        batch_results.append({
                            'case_id': idx + 1,
                            'text': text[:100] + "...",
                            'base_time': base_result.inference_time,
                            'lora_time': lora_result.inference_time,
                            'base_entities': base_result.entities,
                            'lora_entities': lora_result.entities,
                            'comparison': comparison
                        })

                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Base Model", f"{len(base_result.entities)} 实体")
                        with col2:
                            st.metric("LoRA Model", f"{len(lora_result.entities)} 实体")
                        with col3:
                            improvement = comparison['improvement']
                            if improvement > 0:
                                st.metric("改进", f"+{improvement} 实体", delta_color="normal")
                            elif improvement < 0:
                                st.metric("减少", f"{improvement} 实体", delta_color="inverse")
                            else:
                                st.metric("无变化", "0")

                progress_bar.empty()

                # Store batch results
                st.session_state.batch_results = batch_results

                st.success(f"批量测试完成！共测试 {len(test_cases)} 个用例")

                # Show summary statistics
                st.subheader("📈 批量测试汇总")

                total_base_entities = sum(r['comparison']['base_total'] for r in batch_results)
                total_lora_entities = sum(r['comparison']['lora_total'] for r in batch_results)
                avg_base_time = sum(r['base_time'] for r in batch_results) / len(batch_results)
                avg_lora_time = sum(r['lora_time'] for r in batch_results) / len(batch_results)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总实体数 (Base)", total_base_entities)
                with col2:
                    st.metric("总实体数 (LoRA)", total_lora_entities)
                with col3:
                    improvement = total_lora_entities - total_base_entities
                    st.metric("总改进", f"+{improvement}" if improvement > 0 else improvement)
                with col4:
                    time_diff = avg_lora_time - avg_base_time
                    st.metric("平均时间差", f"{time_diff:.2f}s")

else:
    # Single text mode
    st.subheader("🔍 单文本分析模式")

    # Display input text
    if 'input_text' in locals() and input_text:
        st.markdown("#### 输入文本")
        st.info(input_text)

    # Run analysis when button is clicked
    if st.session_state.run_analysis and 'input_text' in locals() and input_text:
        with st.spinner("正在分析中，请稍候..."):
            # Run both models in parallel
            base_result, lora_result = st.session_state.client.extract_entities_both(input_text)

            # Calculate comparison
            comparison = st.session_state.client.compare_entities(
                base_result.entities, lora_result.entities
            )

            st.session_state.test_results = {
                'text': input_text,
                'base': base_result,
                'lora': lora_result,
                'comparison': comparison
            }

# Display results
if st.session_state.test_results:
    results = st.session_state.test_results

    st.markdown("---")

    # Summary statistics
    st.subheader("📊 分析结果摘要")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown("<h3>Base Model</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 2em;'>{len(results['base'].entities)}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>实体数<br>{results['base'].inference_time:.2f}s</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown("<h3>LoRA Model</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 2em;'>{len(results['lora'].entities)}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>实体数<br>{results['lora'].inference_time:.2f}s</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        improvement = len(results['lora'].entities) - len(results['base'].entities)
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown("<h3>改进</h3>", unsafe_allow_html=True)
        color = '#28a745' if improvement >= 0 else '#dc3545'
        sign = '+' if improvement >= 0 else ''
        status = '✓ 改进' if improvement > 0 else ('→ 持平' if improvement == 0 else '↓ 减少')
        st.markdown(f"<p style='font-size: 2em; color: {color};'>{sign}{improvement}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>实体数<br><span class='improvement-badge'>{status}</span></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        time_diff = results['lora'].inference_time - results['base'].inference_time
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown("<h3>时间差</h3>", unsafe_allow_html=True)
        color = '#dc3545' if time_diff > 0 else '#28a745'
        sign = '+' if time_diff > 0 else ''
        status_text = 'LoRA较慢' if time_diff > 0 else ('LoRA较快' if time_diff < 0 else '相同')
        st.markdown(f"<p style='font-size: 2em; color: {color};'>{sign}{time_diff:.2f}s</p>", unsafe_allow_html=True)
        st.markdown(f"<p>推理时间<br>{status_text}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Side-by-side results
    st.subheader("🔄 并排结果对比")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='model-card base-model-card'>", unsafe_allow_html=True)
        st.markdown("<h3>🔵 Base Model (Qwen3-4B)</h3>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>推理时间：</strong> {results['base'].inference_time:.3f} 秒</p>",
                    unsafe_allow_html=True)

        if results['base'].entities:
            st.markdown("<h4>提取的实体：</h4>", unsafe_allow_html=True)
            for entity in results['base'].entities:
                entity_name = entity.get('name', 'N/A')
                entity_type = entity.get('type', 'N/A')
                html_content = f"<p>• <span class='entity-highlight'>{entity_name}</span> <span style='color: #666;'>({entity_type})</span></p>"
                st.markdown(html_content, unsafe_allow_html=True)
        else:
            st.warning("未提取到实体")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='model-card lora-model-card'>", unsafe_allow_html=True)
        st.markdown("<h3>🟣 LoRA Model (Zero3 Fine-tuned)</h3>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>推理时间：</strong> {results['lora'].inference_time:.3f} 秒</p>",
                    unsafe_allow_html=True)

        if results['lora'].entities:
            st.markdown("<h4>提取的实体：</h4>", unsafe_allow_html=True)
            for entity in results['lora'].entities:
                entity_name = entity.get('name', 'N/A')
                entity_type = entity.get('type', 'N/A')
                html_content = f"<p>• <span class='entity-highlight'>{entity_name}</span> <span style='color: #666;'>({entity_type})</span></p>"
                st.markdown(html_content, unsafe_allow_html=True)
        else:
            st.warning("未提取到实体")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Charts section
    st.subheader("📈 可视化图表")

    # Create tabs for different charts
    tab1, tab2, tab3 = st.tabs(["推理速度对比", "实体类型分布", "提取能力雷达图"])

    with tab1:
        chart_html = create_inference_speed_chart(
            {'inference_time': results['base'].inference_time},
            {'inference_time': results['lora'].inference_time}
        )
        # Wrap with plotly.js for rendering
        full_html = f"""
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <div style="height: 450px;">
            {chart_html}
        </div>
        """
        st.components.v1.html(full_html, height=450, scrolling=False)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4>Base Model 实体分布</h4>", unsafe_allow_html=True)
            chart_html = create_entity_type_distribution(results['base'].entities)
            full_html = f"""
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <div style="height: 400px;">
                {chart_html}
            </div>
            """
            st.components.v1.html(full_html, height=400, scrolling=False)
        with col2:
            st.markdown("<h4>LoRA Model 实体分布</h4>", unsafe_allow_html=True)
            chart_html = create_entity_type_distribution(results['lora'].entities)
            full_html = f"""
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <div style="height: 400px;">
                {chart_html}
            </div>
            """
            st.components.v1.html(full_html, height=400, scrolling=False)

    with tab3:
        chart_html = create_comparison_radar_chart(
            results['base'].entities,
            results['lora'].entities
        )
        full_html = f"""
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <div style="height: 500px;">
            {chart_html}
        </div>
        """
        st.components.v1.html(full_html, height=500, scrolling=False)

    st.markdown("---")

    # Detailed comparison table
    st.subheader("📝 实体对比详情")
    comparison_table = create_entity_comparison_table(
        results['base'].entities,
        results['lora'].entities
    )
    st.components.v1.html(comparison_table, height=400, scrolling=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>SFT-ner Model Comparison Demo</strong></p>
    <p style="font-size: 0.9em;">Powered by Qwen3-4B + LoRA ZeRO3 Fine-tuning</p>
</div>
""", unsafe_allow_html=True)
