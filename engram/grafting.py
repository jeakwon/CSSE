
class GraftedModel(nn.Module):
    def __init__(self, base_model, fine_model, init_graft_ratio=0.0001, sigmoid_bias=-5, device="cuda:0"):
        super(GraftedModel, self).__init__()

        self.base_model = deepcopy(base_model).train()
        self.fine_model = deepcopy(fine_model).train()
        self.base_state_dict = {k: v.clone().detach() for k, v in base_model.state_dict().items()}
        self.fine_state_dict = {k: v.clone().detach() for k, v in fine_model.state_dict().items()}
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

        self.init_graft_ratio = init_graft_ratio
        self.graft_ratio = init_graft_ratio
        self.sigmoid_bias = sigmoid_bias
        self.device = device

        self.param_diffs = self.get_param_diffs()
        self.base_masks = self.get_base_masks()
        self.trainable_params = self.get_trainable_params()
        self.is_hard_mask_on = False

    def get_param_diffs(self):
        return {name:
                self.fine_state_dict[name] -
                self.base_state_dict[name]
                for name in self.base_state_dict.keys()}

    def get_base_masks(self):
        all_diffs = torch.cat([p.flatten() for p in self.param_diffs.values()])
        threshold = torch.quantile(torch.abs(all_diffs), 1.0 - self.graft_ratio)
        return {name:
                (torch.abs(param_diff) >= threshold).int()
                for name, param_diff in self.param_diffs.items()}

    def get_trainable_params(self):
        trainable_params = {name:
                            nn.Parameter(torch.randn_like(param.float(), requires_grad=True))
                            for name, param in self.base_state_dict.items()}
        self._trainable_params = nn.ParameterList(trainable_params.values())  # Register
        return trainable_params

    @property
    def soft_masks(self):
        soft_masks = {}
        for name, param in self.trainable_params.items():
            m = self.base_masks[name]
            s = torch.sigmoid(param + self.sigmoid_bias)

            soft_mask = (1-m) * s + m * (1-s)
            soft_masks[name] = soft_mask
        return soft_masks

    @property
    def hard_masks(self):
        masked_diffs = { name: torch.abs( diff * mask )
        for (name, diff), (_, mask) in zip(self.param_diffs.items(), self.soft_masks.items() ) }

        all_diffs = torch.cat([p.flatten() for p in masked_diffs.values()])
        threshold = torch.quantile(torch.abs(all_diffs), 1.0 - self.graft_ratio)

        hard_masks = {}
        for name, masked_diff in masked_diffs.items():
            hard_mask = (masked_diff>=threshold).int()
            hard_masks[name] = hard_mask
        return hard_masks

    def get_grafted_params(self, masks):
        return {name:
                self.fine_state_dict[name] * masks[name] +
                self.base_state_dict[name] * (1-masks[name])
                for name in self.base_state_dict.keys()}

    def forward(self, x):
        if self.is_hard_mask_on:
            grafted_params = self.get_grafted_params(self.hard_masks)
        else:
            grafted_params = self.get_grafted_params(self.soft_masks)

        return torch.func.functional_call(self.base_model, grafted_params, (x,))

    def report_status(self):
        flatten_params =  torch.cat([p.flatten() for p in self.trainable_params.values()])
        avg_params = flatten_params.mean()

        flatten_soft_masks =  torch.cat([p.flatten() for p in self.soft_masks.values()])
        flatten_hard_masks =  torch.cat([p.flatten() for p in self.hard_masks.values()])
        soft_mask_mean = flatten_soft_masks.sum()/len(flatten_soft_masks)
        hard_mask_mean = flatten_hard_masks.sum()/len(flatten_hard_masks)
        print(f"Average Param Value : {avg_params}, Soft Mask Mean : {soft_mask_mean}, Hard Mask Mean : {hard_mask_mean}")

    def apply_hard_mask_graft_ratio(self, graft_ratio=None):
        if graft_ratio is not None:
            self.graft_ratio = graft_ratio
            self.is_hard_mask_on = True
        else:
            self.graft_ratio = self.init_graft_ratio
            self.is_hard_mask_on = False