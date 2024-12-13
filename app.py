import streamlit as st
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델 및 인코더 로드
kmbert_tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424")
num_labels = len(joblib.load("checkpoint/model_filelabel_encoder_diagnosis.pkl").classes_)  # 진료과목 라벨 개수 로드
kmbert_model = AutoModelForSequenceClassification.from_pretrained("madatnlp/km-bert", num_labels=num_labels)
kmbert_model.load_state_dict(torch.load("/Users/ham-yanghun/Desktop/University/University/24-2/ML/final_project/checkpoint/model_filekmbert_finetuned_model.pt", map_location='cpu'))
kmbert_model.eval()

stack_model_1 = joblib.load("/Users/ham-yanghun/Desktop/University/University/24-2/ML/final_project/checkpoint/model_filestack_model_1.pkl")           # 1차 분류 스택 모델
stack_model_2 = joblib.load("/Users/ham-yanghun/Desktop/University/University/24-2/ML/final_project/checkpoint/model_filestack_model_2.pkl")           # 2차 분류 모델
xgb_model = joblib.load("/Users/ham-yanghun/Desktop/University/University/24-2/ML/final_project/checkpoint/model_filexgb_model_for_1st_stage.pkl")     # 1차 분류용 XGBoost 모델

label_encoder_diagnosis = joblib.load("checkpoint/model_filelabel_encoder_diagnosis.pkl")
label_encoder_code = joblib.load("/Users/ham-yanghun/Desktop/University/University/24-2/ML/final_project/checkpoint/model_filelabel_encoder_code.pkl")
one_hot_encoder = joblib.load("/Users/ham-yanghun/Desktop/University/University/24-2/ML/final_project/checkpoint/model_fileonehot_encoder.pkl")
scaler = joblib.load("/Users/ham-yanghun/Desktop/University/University/24-2/ML/final_project/checkpoint/model_filescaler.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kmbert_model.to(device)

def predict_kmbert_proba(text):
    tokens = kmbert_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    with torch.no_grad():
        outputs = kmbert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

st.title("의료 데이터 예측 데모")

# 사용자 입력받기
symptom = st.text_input("증상을 입력하세요", "머리가 아프고, 복통과 설사가 있어요")
age = st.number_input("나이를 입력하세요", min_value=0, max_value=120, value=18)
sex_choice = st.selectbox("성별을 선택하세요", options=[1,2], format_func=lambda x: "남자" if x==1 else "여자")
in_day = st.number_input("입내원일수", min_value=0, value=0)
pres_day = st.number_input("총처방일수", min_value=0, value=0)
cure_day = st.number_input("요양일수", min_value=0, value=2)

if st.button("예측하기"):
    # KM-BERT 예측 확률
    kmbert_probs = predict_kmbert_proba(symptom)

    # 정형데이터 처리
    # 주의: 연령대코드 계산 로직은 학습 시 사용했던 로직에 맞춰야 함
    # 여기서는 단순히 나이//10을 연령대코드로 가정 (학습과정과 동일해야 함)
    age_code = age // 10

    X_input = np.array([[sex_choice, age_code, cure_day, in_day, pres_day]])
    X_input_enc = one_hot_encoder.transform(X_input)
    X_input_scaled = scaler.transform(X_input_enc)

    # 1차분류용 XGB 모델 예측 확률
    xgb_1_proba = xgb_model.predict_proba(X_input_scaled)

    # 스택 1차 분류 입력
    stack_input_1 = np.hstack([kmbert_probs, xgb_1_proba])
    first_stage_pred = stack_model_1.predict(stack_input_1)
    first_stage_pred_label = label_encoder_diagnosis.inverse_transform(first_stage_pred)[0]

    # 2차 분류 입력: 테스트 시 1차 예측결과(first_stage_pred)를 사용
    stack_input_2 = np.hstack([stack_input_1, first_stage_pred.reshape(-1,1)])
    second_stage_pred = stack_model_2.predict(stack_input_2)
    second_stage_pred_label = label_encoder_code.inverse_transform(second_stage_pred)[0]

    # 출력: 2차 분류 결과를 먼저, 그 다음 1차 분류 결과
    st.write(f"당신은 {second_stage_pred_label}(이)가 의심됩니다. 가까운 {first_stage_pred_label}(을/를) 방문하세요.")
