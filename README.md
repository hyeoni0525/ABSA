# 💄 Aspect-Based Sentiment Analysis (ABSA) on Cosmetics Reviews

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://github.com/huggingface/peft)

> **"화장품 리뷰의 숨은 의도까지 파악한다."**
> 11가지 속성(보습, 가격, 향 등)을 탐지하고, 각 속성별 감성(긍정/부정)을 정교하게 추출하는 BERT 기반 NLP 프로젝트입니다.

---

## 1. Project Overview

단순한 긍/부정 분류를 넘어, 하나의 문장 안에 섞여 있는 복합적인 감정을 속성별로 분리하여 비즈니스 인사이트를 도출하는 것을 목표로 했습니다.

* **기간:** 202X.XX ~ 202X.XX (2일)
* **역할:** AI 모델링 및 파인튜닝 (개인 및 조별 프로젝트)
* **데이터:** 뷰엘라 코스메틱스 리뷰 데이터 약 10만 건 (Text, 11 Aspect, Label)
* **목표:** "가격은 비싸지만(부정), 보습은 좋다(긍정)"와 같은 복합 리뷰 분석 시스템 구축

---

## 2. Background & Problem

### 🧐 Why ABSA?
기존의 쇼핑몰 리뷰 분석은 단순히 별점이나 전체적인 긍/부정만을 다루었습니다. 하지만 화장품 도메인 특성상 고객은 **"보습력", "밀착력", "향", "가격"** 등 다양한 속성(Aspect)에 대해 서로 다른 감정을 하나의 문장에서 표현합니다.

> **Problem:** "상품이 촉촉하고 좋은데(보습:긍정), 양이 적어요(용량:부정)"라는 리뷰를 단순히 '긍정'이나 '부정' 하나로 분류하면 데이터의 가치가 손실됩니다.
>
> **Solution:** 문장 내 속성을 감지하고 각 속성의 감성을 별도로 분석하는 **ABSA(Aspect-Based Sentiment Analysis)** 시스템을 구축했습니다.

---

## 3. Tech Stack

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Language** | Python | Main Programming |
| **Library** | PyTorch, Hugging Face | Model Training & Inference |
| **Optimization** | PEFT (LoRA) | Efficient Fine-tuning |
| **Model** | `klue/bert-base`<br>`klue/roberta-base` | Pre-trained Korean Language Models |

---

## 4. Methodology & Process 🔥

복잡한 문제를 해결하기 위해 **Divide and Conquer (단계별 정복)** 전략을 사용하여 모델을 파이프라인화 했습니다.

### Step 1: Baseline (Binary Classification)
* 리뷰 전체의 긍/부정을 판단하는 기본 모델 생성
* `klue/bert-base` 모델을 사용하여 한국어 리뷰 데이터에 대한 **Domain Adaptation** 수행

### Step 2: Aspect Detection (Multi-label Classification)
하나의 리뷰에 '보습', '향', '가격' 등 여러 속성이 동시에 등장하는 문제를 해결합니다.

* **Action:** 11개 속성에 대한 존재 여부를 예측하는 **Multi-label Classification** 문제로 정의
* **Implementation:**
    * Hugging Face Trainer의 `problem_type="multi_label_classification"` 설정
    * `BCEWithLogitsLoss` 계산을 위해 Label을 Float로 변환하는 **Custom Data Collator** 구현

### Step 3: Aspect-Based Sentiment Analysis (ABSA)
특정 속성(예: 향)에 대해서만 긍/부정을 판단하도록 모델을 학습시킵니다.

* **Strategy:** 입력 데이터 구조 변경을 통해 모델의 Attention 유도
    * **Case 1:** Special Token 활용 `[ASPECT] 속성 [SEP] 리뷰문장`
    * **Case 2:** 텍스트 프롬프트 활용 `속성 : 리뷰문장`
* **Token Handling:** `tokenizer.add_special_tokens`로 `[ASPECT]` 토큰 추가 및 `model.resize_token_embeddings` 적용

### 🚀 Efficiency: LoRA (Low-Rank Adaptation)
* 대규모 언어 모델의 모든 파라미터를 학습시키는 대신, **LoRA**를 적용하여 학습 파라미터 수를 획기적으로 줄임
* `LoraConfig`를 통해 Rank(r=8) 설정 후 빠른 실험 반복 수행

---

## 5. Troubleshooting

개발 과정에서 발생한 주요 이슈와 해결 과정입니다.

* **Issue:** Hugging Face Trainer 사용 시 손실 함수 계산 중 `RuntimeError` 발생
* **Cause:** Multi-label classification에서 사용하는 `BCEWithLogitsLoss`는 Target(Label)이 Float 타입이어야 함 (기본 로더는 Long 타입 반환)
* **Solution:** 배치 생성 시 강제로 타입을 변환하는 커스텀 콜레이터를 구현하여 해결

```python
def float_label_collator(features):
    batch = default_data_collator(features)
    batch["labels"] = batch["labels"].float()  # 핵심: Float 변환
    return batch
```
## 6. Result & Impact

### 📊 Performance
*(아래 수치는 예시입니다. 실제 수치로 변경해주세요.)*

* **속성 검출 모델 F1-Score:** 0.XX (Macro Average)
* **ABSA 모델 정확도:** 0.XX

### 📈 Visualization
*(PDF의 속성별 F1-Score 막대그래프 캡처 이미지를 여기에 넣어주세요)*

![F1 Score Chart](path/to/chart_image.png)

### 💼 Business Impact
* 단순 별점보다 구체적인 **VOC(Voice of Customer)** 분석 가능
* "가격 불만" 고객과 "품질 만족" 고객을 세분화하여 **타겟 마케팅 전략 수립** 가능

---

## 7. Retrospective

* **Learned:** Multi-label 데이터셋 처리 방법과 Special Token을 추가하여 모델의 입력 구조를 제어하는 기술을 익혔습니다. 특히 LoRA를 통해 적은 리소스로도 LLM을 효과적으로 튜닝할 수 있음을 확인했습니다.
* **Future Work:** 데이터 불균형(Imbalance) 문제로 일부 속성의 예측력이 낮았습니다. 향후에는 **Focal Loss**를 도입하거나 **Data Augmentation**을 통해 소수 클래스의 성능을 보완할 계획입니다.

---

## 8. How to Run

```bash
# Clone the repository
git clone [https://github.com/your-username/absa-project.git](https://github.com/your-username/absa-project.git)

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --model klue/bert-base --method lora
