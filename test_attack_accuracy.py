import torch
import numpy as np
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Test attack accuracy.")
    parser.add_argument("--target_model_member_path", required=True)
    parser.add_argument("--target_model_non_member_path", required=True)
    parser.add_argument("--shadow_model_member_path", nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument("--shadow_model_non_member_path", nargs='+', help='<Required> Set flag', required=True)

    return parser.parse_args()


def load_shadow_data():
    shadow_model_member_list = []
    shadow_model_non_member_list = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for shadow_mem, shadow_non_mem in zip(args.shadow_model_member_path, args.shadow_model_non_member_path):
        shadow_model_member_list.append(torch.load(shadow_mem).to(device))
        shadow_model_non_member_list.append(torch.load(shadow_non_mem).to(device))
    shadow_member = torch.cat(shadow_model_member_list, dim = 0)
    shadow_non_member = torch.cat(shadow_model_non_member_list, dim = 0)
    return shadow_member, shadow_non_member

def load_target_data(member_path, non_member_path):
    target_member = torch.load(member_path)
    target_non_member = torch.load(non_member_path)
    return target_member, target_non_member

def preprocess(member, non_member):
    train_np = member.cpu().numpy()
    test_np = non_member.cpu().numpy()
    train_np = train_np[0:test_np.shape[0]]
    train_y_np = np.ones(train_np.shape[0])
    test_y_np = np.zeros(test_np.shape[0])
    x = np.vstack((train_np, test_np))
    y = np.concatenate((train_y_np, test_y_np))
    x = preprocessing.scale(x)
    return x, y

if __name__ == "__main__":
    args = parse_args()
    shadow_member, shadow_non_member = load_shadow_data()
    shadow_x, shadow_y = preprocess(shadow_member, shadow_non_member)
    shadow_train_x, shadow_test_x, shadow_train_y, shadow_test_y = train_test_split(shadow_x, shadow_y, test_size = 0.3)
    print("Training XGB...")
    xgb = XGBClassifier(n_estimators=200)
    xgb.fit(shadow_train_x, shadow_train_y)
    pred_xgb = xgb.predict(shadow_train_x)
    print("Shadow Train Results -------------------------------")
    print("XGBoost Classification Report=\n\n", classification_report(shadow_train_y, pred_xgb)) 
    print("XGBoost Confusion Matrix=\n\n", confusion_matrix(shadow_train_y, pred_xgb)) 

    pred_xgb = xgb.predict(shadow_test_x)
    print("Shadow Test Results -------------------------------")
    print("XGBoost Classification Report=\n\n", classification_report(shadow_test_y, pred_xgb)) 
    print("XGBoost Confusion Matrix=\n\n", confusion_matrix(shadow_test_y, pred_xgb)) 
    
    target_member, target_non_member = load_target_data(args.target_model_member_path, args.target_model_non_member_path)
    target_x, target_y = preprocess(target_member, target_non_member)
    pred_xgb = xgb.predict(target_x)
    print("Target Attack Results -------------------------------")
    print("XGBoost Classification Report=\n\n", classification_report(target_y, pred_xgb,digits = 3)) 
    print("XGBoost Confusion Matrix=\n\n", confusion_matrix(target_y, pred_xgb))

    pred_xgb = xgb.predict_proba(target_x)
    roc_auc = roc_auc_score(target_y, pred_xgb[:,1])

    print(f"ROC AUC: {roc_auc}")
    
    fpr, tpr, _ = roc_curve(target_y, pred_xgb[:,1])

    desired_fpr = 0.001

    closest_fpr_index = np.argmin(np.abs(fpr - desired_fpr))
    tpr_at_desired_fpr = tpr[closest_fpr_index]

    print(f"TPR at FPR = {desired_fpr}: {tpr_at_desired_fpr}")