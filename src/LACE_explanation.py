class LACE_explanation:
    """The LACE_explainer class, through the `explain_instance` method allows
    to obtain a rule-based model-agnostic local explanation for an instance.

    Parameters
    ----------
    clf : sklearn classifier
        Any sklearn-like classifier can be passed. It must have the methods
        `predict` and `predict_proba`.
    train_dataset : Dataset
        The dataset from which the locality of the explained instance is created.
    min_sup : float
        L^3 Classifier's Minimum support parameter.
    """

    def __init__(self, LACE_explainer_o, diff_single, map_difference, k, error, instance, target_class, errors, instance_class_index, prob, metas=None):
        self.LACE_explainer_o=LACE_explainer_o
        self.diff_single=diff_single
        self.map_difference=map_difference
        self.k=k
        self.error=error
        self.instance=instance
        self.target_class=target_class
        self.errors=errors
        self.instance_class_index=instance_class_index
        self.prob=prob
        self.metas=metas
        self.infos={"d":self.LACE_explainer_o.dataset_name, "model":self.LACE_explainer_o.model, "x":self.metas}#d_explain.metas[i]}



    def plotExplanation(self):
        import matplotlib.pyplot as plt
        #plt.style.use('seaborn-talk')
        
        fig, pred_ax = plt.subplots(1, 1)
        
        attrs_values = [f"{k}={v}" for k,v in (self.instance.items())][:-1]
        infos_t=" ".join([f"{k}={self.infos[k]}" if k in self.infos and self.infos[k] is not None else "" for k in ["x", "d", "model"]])
        #" ".join([f"{k}={v}" for k, v in infos.items()])
        title = f"{infos_t}\n p(class={self.target_class}|x)={self.prob:.2f} true class={self.instance.iloc[-1]}"    

        pred_ax.set_title(title)
        pred_ax.set_xlabel(f"Δ target class={self.instance.iloc[-1]}")
        
        
        dict_instance=dict(enumerate([f"{k}={v}" for k,v in self.instance.to_dict().items()], start=1))
        rules={", ".join([dict_instance[int(i)] for i in rule_i.split(",")]):v for rule_i,v in self.map_difference.items()}
        mapping_rules={list(rules.keys())[i]:f"Rule_{i+1}" for i in range(0, len(rules))}
        #Do not plot rules of 1 item
        rules_plot={mapping_rules[k]:v for k,v in rules.items() if len(k.split(", "))>1}
        pred_ax.barh(
            attrs_values + list(rules_plot.keys()),
            width=self.diff_single + list(rules_plot.values()),
            align='center', color="#bee2e8", linewidth='1', edgecolor='black'
        )
        pred_ax.invert_yaxis()
        print([f"{v}={{{k}}}" for k,v in mapping_rules.items()])
        #fig.show()
        #plt.close()
        