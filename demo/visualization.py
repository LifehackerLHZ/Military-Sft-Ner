"""
Visualization components for NER Model Comparison
Charts and statistical visualizations using Plotly
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List


def create_inference_speed_chart(base_result: Dict[str, Any],
                                 lora_result: Dict[str, Any]) -> str:
    """
    Create a bar chart comparing inference speeds

    Args:
        base_result: Base model result
        lora_result: LoRA model result

    Returns:
        HTML div string for the chart
    """
    data = {
        'Model': ['Base Model', 'LoRA Model'],
        'Inference Time (s)': [base_result.get('inference_time', 0),
                               lora_result.get('inference_time', 0)]
    }

    fig = go.Figure(data=[
        go.Bar(
            x=data['Model'],
            y=data['Inference Time (s)'],
            marker_color=['#636EFA', '#EF553B'],
            text=[f"{t:.2f}s" for t in data['Inference Time (s)']],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="模型推理速度对比",
        xaxis_title="模型",
        yaxis_title="推理时间 (秒)",
        height=400,
        font=dict(size=12),
        showlegend=False
    )

    return fig.to_html(div_id="inference-speed-chart", include_plotlyjs=False)


def create_entity_type_distribution(entities: List[Dict[str, Any]]) -> str:
    """
    Create a pie chart showing entity type distribution

    Args:
        entities: List of entities

    Returns:
        HTML div string for the chart
    """
    # Define color mapping for different entity types
    color_mapping = {
        '军事装备': '#EF553B',  # Red
        '地理位置': '#00CC96',  # Green
        '组织名称': '#636EFA',  # Blue
        '人名': '#FFA15A',      # Orange
        '未知': '#B6B6B6'       # Gray
    }

    type_counts = {}
    for entity in entities:
        etype = entity.get('type', '未知')
        type_counts[etype] = type_counts.get(etype, 0) + 1

    if not type_counts:
        return "<p>未提取到实体</p>"

    labels = list(type_counts.keys())
    values = list(type_counts.values())

    # Map colors based on entity types
    colors = [color_mapping.get(label, '#B6B6B6') for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=colors,
        textinfo='label+percent',  # Show label and percentage
        textposition='outside',    # Place labels outside the pie
        textfont=dict(size=11),    # Set font size for labels
        showlegend=True            # Show legend for better readability
    )])

    # Adjust margins to accommodate labels
    fig.update_layout(
        title="实体类型分布",
        height=400,
        font=dict(size=12),
        margin=dict(t=80, b=20, l=20, r=20),  # Adjust margins
        annotations=[dict(text='实体类型', x=0.5, y=0.5, font_size=12, showarrow=False)]
    )

    return fig.to_html(div_id="entity-distribution-chart", include_plotlyjs=False)


def create_comparison_radar_chart(base_entities: List[Dict[str, Any]],
                                  lora_entities: List[Dict[str, Any]]) -> str:
    """
    Create a radar chart comparing entity extraction across types

    Args:
        base_entities: Base model entities
        lora_entities: LoRA model entities

    Returns:
        HTML div string for the chart
    """
    entity_types = ['军事装备', '地理位置', '组织名称', '人名']

    def count_by_type(entities):
        counts = {t: 0 for t in entity_types}
        for entity in entities:
            etype = entity.get('type', '')
            if etype in counts:
                counts[etype] += 1
        return [counts[t] for t in entity_types]

    base_counts = count_by_type(base_entities)
    lora_counts = count_by_type(lora_entities)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=base_counts,
        theta=entity_types,
        fill='toself',
        name='Base Model',
        line_color='rgb(99, 110, 250)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=lora_counts,
        theta=entity_types,
        fill='toself',
        name='LoRA Model',
        line_color='rgb(239, 85, 59)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(base_counts), max(lora_counts)) + 2]
            ),
            angularaxis=dict(
                tickfont=dict(size=12),  # Set font size for entity type labels
                rotation=90  # Rotate to prevent overlap
            )
        ),
        showlegend=True,
        title="实体类型提取能力对比 (雷达图)",
        height=500,
        font=dict(size=12),
        legend=dict(
            x=0.8,  # Position legend to the right
            y=0.5,
            font=dict(size=11)
        )
    )

    return fig.to_html(div_id="radar-chart", include_plotlyjs=False)


def create_batch_performance_chart(batch_results: List[Dict[str, Any]]) -> str:
    """
    Create a chart showing batch evaluation performance

    Args:
        batch_results: List of batch evaluation results

    Returns:
        HTML div string for the chart
    """
    if not batch_results:
        return "<p>暂无批量测试结果</p>"

    test_cases = [f"Case {i + 1}" for i in range(len(batch_results))]

    base_times = [r.get('base_time', 0) for r in batch_results]
    lora_times = [r.get('lora_time', 0) for r in batch_results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=test_cases,
        y=base_times,
        name='Base Model',
        marker_color='#636EFA',
        text=[f"{t:.2f}s" for t in base_times],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x=test_cases,
        y=lora_times,
        name='LoRA Model',
        marker_color='#EF553B',
        text=[f"{t:.2f}s" for t in lora_times],
        textposition='auto'
    ))

    fig.update_layout(
        title="批量测试推理时间对比",
        xaxis_title="测试用例",
        yaxis_title="推理时间 (秒)",
        barmode='group',
        height=500,
        font=dict(size=12)
    )

    return fig.to_html(div_id="batch-performance-chart", include_plotlyjs=False)


def create_metrics_comparison_chart(metrics: Dict[str, Any]) -> str:
    """
    Create a chart comparing precision, recall, F1 scores

    Args:
        metrics: Dictionary with base and lora metrics

    Returns:
        HTML div string for the chart
    """
    if 'base' not in metrics or 'lora' not in metrics:
        return "<p>暂无指标数据</p>"

    metrics_names = ['Precision', 'Recall', 'F1 Score']
    base_values = [
        metrics['base'].get('precision', 0) * 100,
        metrics['base'].get('recall', 0) * 100,
        metrics['base'].get('f1', 0) * 100
    ]
    lora_values = [
        metrics['lora'].get('precision', 0) * 100,
        metrics['lora'].get('recall', 0) * 100,
        metrics['lora'].get('f1', 0) * 100
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics_names,
        y=base_values,
        name='Base Model',
        marker_color='#636EFA',
        text=[f"{v:.1f}%" for v in base_values],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x=metrics_names,
        y=lora_values,
        name='LoRA Model',
        marker_color='#EF553B',
        text=[f"{v:.1f}%" for v in lora_values],
        textposition='auto'
    ))

    fig.update_layout(
        title="模型评估指标对比",
        xaxis_title="评估指标",
        yaxis_title="数值 (%)",
        barmode='group',
        height=450,
        font=dict(size=12)
    )

    return fig.to_html(div_id="metrics-chart", include_plotlyjs=False)


def create_entity_comparison_table(base_entities: List[Dict[str, Any]],
                                   lora_entities: List[Dict[str, Any]]) -> str:
    """
    Create a table showing entity comparison between models

    Args:
        base_entities: Base model entities
        lora_entities: LoRA model entities

    Returns:
        HTML table string
    """
    def entity_key(entity):
        name = entity.get("name", "")
        type_ = entity.get("type", "")
        return f"{name}|{type_}"

    base_set = {entity_key(e) for e in base_entities}
    lora_set = {entity_key(e) for e in lora_entities}

    common = base_set & lora_set
    base_only = base_set - lora_set
    lora_only = lora_set - base_set

    # Create table
    html = """
    <div style="margin-top: 20px;">
        <h4>实体对比详情</h4>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="background-color: #f0f0f0;">
                    <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">模型</th>
                    <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">实体数量</th>
                    <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">实体示例</th>
                </tr>
            </thead>
            <tbody>
    """

    # Base model only
    if base_only:
        html += f"""
                <tr style="background-color: #ffffff;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Base Model Only</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{len(base_only)}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">
                        {', '.join(list(base_only)[:5])}{'...' if len(base_only) > 5 else ''}
                    </td>
                </tr>
        """

    # LoRA model only (improvements)
    if lora_only:
        html += f"""
                <tr style="background-color: #fff3cd;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>LoRA Model Only ✨</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{len(lora_only)}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">
                        {', '.join(list(lora_only)[:5])}{'...' if len(lora_only) > 5 else ''}
                    </td>
                </tr>
        """

    # Common entities
    if common:
        html += f"""
                <tr style="background-color: #d4edda;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>共同提取</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{len(common)}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">
                        {', '.join(list(common)[:5])}{'...' if len(common) > 5 else ''}
                    </td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    """

    return html
