from sklearn.metrics import fbeta_score, cohen_kappa_score, balanced_accuracy_score

def compute_test_metrics(real_target, predicted_target, metric='balanced_accuracy'):
        if metric == 'balanced_accuracy':
            return balanced_accuracy_score(real_target, predicted_target)
        elif metric == 'fb_score':
            return fbeta_score(real_target, predicted_target, beta=1.5, average='weighted')
        elif metric == 'cohen_kappa':
            return cohen_kappa_score(real_target, predicted_target)