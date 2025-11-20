#!/usr/bin/env python3
"""
NER Model Comparison Module
Parallel API calls to Base and LoRA models for side-by-side comparison
"""

import json
import time
import requests
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelResult:
    """Result from a single model inference"""
    model_name: str
    entities: List[Dict[str, Any]]
    inference_time: float
    success: bool
    error_message: str = ""
    raw_response: str = ""


class NERComparisonClient:
    """Parallel NER API client for comparing Base and LoRA models"""

    def __init__(self, base_api_url: str = "http://localhost:8001",
                 lora_api_url: str = "http://localhost:8002"):
        """
        Initialize comparison client

        Args:
            base_api_url: Base model API URL
            lora_api_url: LoRA model API URL
        """
        self.base_api_url = f"{base_api_url}/v1/chat/completions"
        self.lora_api_url = f"{lora_api_url}/v1/chat/completions"

    def format_ner_prompt(self, text: str) -> str:
        """Format text into NER prompt"""
        return f"""你是一个文本实体抽取领域的专家，你需要从给定的句子中提取出实体并且以 json 格式输出, 如 {{"entities": [{{"name":"外层抗击区临界线","type":"军事装备"}}]}}

注意:
1. 输出的每一行都必须是正确的 json 字符串
2. 找不到任何实体时, 输出"没有找到任何实体和关系"
3. 如果地理实体有坐标需要输出地理实体的坐标，例如兰州(36.06,103.79)，没有坐标则输出地理实体
4. 实体类型必须从以下四种实体类型进行选择：军事装备，地理位置，组织名称，人名

输入文本：
{text}

请直接输出JSON结果："""

    def extract_entities_single(self, text: str, api_url: str,
                                model_name: str = "qwen3") -> ModelResult:
        """
        Call single model API to extract entities

        Args:
            text: Input text
            api_url: API endpoint URL
            model_name: Model name

        Returns:
            ModelResult object
        """
        prompt = self.format_ner_prompt(text)

        # Optimized parameters for Qwen3
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 6144,
            "presence_penalty": 1.5,
            "stop": ["\n\n输入文本：", "\n输入文本：", "输入文本："]
        }

        start_time = time.time()

        try:
            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            inference_time = time.time() - start_time

            # Extract entities from response
            entities = self._extract_entities_from_response(response_text)

            return ModelResult(
                model_name=model_name,
                entities=entities,
                inference_time=inference_time,
                success=True,
                raw_response=response_text
            )

        except Exception as e:
            inference_time = time.time() - start_time
            return ModelResult(
                model_name=model_name,
                entities=[],
                inference_time=inference_time,
                success=False,
                error_message=str(e)
            )

    def _extract_entities_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract entities from model response with improved parsing"""
        entities = []

        try:
            # Remove <think> tags and their content if present
            clean_text = response_text
            if '<think>' in clean_text:
                # Extract content after </think> if it exists
                think_end = clean_text.find('</think>')
                if think_end >= 0:
                    clean_text = clean_text[think_end + 8:]

            # Find all JSON objects with entities in the response
            import re
            json_pattern = r'\{\s*"entities"\s*:\s*\[([^\]]*)\]\s*\}'
            matches = re.findall(json_pattern, clean_text, re.DOTALL)

            if matches:
                # Reconstruct and parse the first valid JSON object
                for match in matches:
                    try:
                        json_str = f'{{"entities": [{match}]}}'
                        data = json.loads(json_str)
                        if "entities" in data and isinstance(data["entities"], list):
                            entities.extend(data["entities"])
                    except json.JSONDecodeError:
                        continue

            # If no entities found yet, try direct JSON parsing of the entire response
            if not entities:
                lines = clean_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            data = json.loads(line)
                            if "entities" in data:
                                return data["entities"]
                        except json.JSONDecodeError:
                            continue

            # Return unique entities
            seen = set()
            unique_entities = []
            for entity in entities:
                key = f"{entity.get('name', '')}|{entity.get('type', '')}"
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)

            return unique_entities

        except Exception:
            pass

        return entities

    def extract_entities_both(self, text: str) -> Tuple[ModelResult, ModelResult]:
        """
        Extract entities from both models in parallel

        Args:
            text: Input text

        Returns:
            Tuple of (base_result, lora_result)
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both API calls
            base_future = executor.submit(
                self.extract_entities_single,
                text,
                self.base_api_url,
                "qwen3-base"
            )

            lora_future = executor.submit(
                self.extract_entities_single,
                text,
                self.lora_api_url,
                "qwen3-ner-zero3"
            )

            # Get results
            base_result = base_future.result()
            lora_result = lora_future.result()

        return base_result, lora_result

    def compare_entities(self, base_entities: List[Dict[str, Any]],
                        lora_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare entities extracted by both models

        Args:
            base_entities: Entities from base model
            lora_entities: Entities from LoRA model

        Returns:
            Comparison statistics
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

        return {
            "base_total": len(base_entities),
            "lora_total": len(lora_entities),
            "common": len(common),
            "base_only": len(base_only),
            "lora_only": len(lora_only),
            "improvement": len(lora_entities) - len(base_entities)
        }

    def calculate_metrics(self, ground_truth: Optional[List[Dict[str, Any]]],
                         predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate precision, recall, F1 score (if ground truth available)

        Args:
            ground_truth: Ground truth entities
            predictions: Predicted entities

        Returns:
            Metrics dictionary
        """
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        def entity_key(entity):
            name = entity.get("name", "")
            type_ = entity.get("type", "")
            return f"{name}|{type_}"

        gt_set = {entity_key(e) for e in ground_truth}
        pred_set = {entity_key(e) for e in predictions}

        true_positives = len(gt_set & pred_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
