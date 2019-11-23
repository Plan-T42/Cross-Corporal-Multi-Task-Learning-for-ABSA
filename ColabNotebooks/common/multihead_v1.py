import torch.nn as nn


class MultiheadV1(nn.Module):
    def __init__(self, aspect_classes, polarity_classes, joint_classes, embedding_dim=768, drop_prob=0.5):
        super(MultiheadV1, self).__init__()

        self.backbone_dims = [embedding_dim // 4, embedding_dim // 4]

        # backbone
        self.backbone = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=self.backbone_dims[0]),
            nn.LayerNorm(self.backbone_dims[0]),
            nn.ELU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=self.backbone_dims[0], out_features=self.backbone_dims[1]),
            nn.LayerNorm(self.backbone_dims[1]),
            nn.ELU()
        )

        # tripadvisor head
        # no sigmoid -> returns logits for torch.nn.BCEWithLogitsLoss()
        self.trip_aspect = nn.Linear(in_features=self.backbone_dims[1], out_features=aspect_classes['trip'])
        self.trip_polarity = nn.Linear(in_features=self.backbone_dims[1], out_features=polarity_classes['trip'])
        self.trip_joint = nn.Linear(in_features=self.backbone_dims[1], out_features=joint_classes['trip'])

        # organic head
        self.organic_aspect = nn.Linear(in_features=self.backbone_dims[1], out_features=aspect_classes['organic'])
        self.organic_polarity = nn.Linear(in_features=self.backbone_dims[1], out_features=polarity_classes['organic'])
        self.organic_joint = nn.Linear(in_features=self.backbone_dims[1], out_features=joint_classes['organic'])

        # SemEval14res head
        self.SemEval14res_aspect = nn.Linear(in_features=self.backbone_dims[1], out_features=aspect_classes['SemEval14res'])
        self.SemEval14res_polarity = nn.Linear(in_features=self.backbone_dims[1], out_features=polarity_classes['SemEval14res'])
        self.SemEval14res_joint = nn.Linear(in_features=self.backbone_dims[1], out_features=joint_classes['SemEval14res'])

        # SemEval14lap head
        self.SemEval14lap_aspect = nn.Linear(in_features=self.backbone_dims[1], out_features=aspect_classes['SemEval14lap'])
        self.SemEval14lap_polarity = nn.Linear(in_features=self.backbone_dims[1], out_features=polarity_classes['SemEval14lap'])
        self.SemEval14lap_joint = nn.Linear(in_features=self.backbone_dims[1], out_features=joint_classes['SemEval14lap'])

        # SemEval16res head
        self.SemEval16res_aspect = nn.Linear(in_features=self.backbone_dims[1], out_features=aspect_classes['SemEval16res'])
        self.SemEval16res_polarity = nn.Linear(in_features=self.backbone_dims[1], out_features=polarity_classes['SemEval16res'])
        self.SemEval16res_joint = nn.Linear(in_features=self.backbone_dims[1], out_features=joint_classes['SemEval16res'])

        # SemEval16lap head
        self.SemEval16lap_aspect = nn.Linear(in_features=self.backbone_dims[1], out_features=aspect_classes['SemEval16lap'])
        self.SemEval16lap_polarity = nn.Linear(in_features=self.backbone_dims[1], out_features=polarity_classes['SemEval16lap'])
        self.SemEval16lap_joint = nn.Linear(in_features=self.backbone_dims[1], out_features=joint_classes['SemEval16lap'])


    def forward(self, input):
        x = self.backbone(input)

        # tripadivisor head
        trip_aspect = self.trip_aspect(x)
        trip_polarity = self.trip_polarity(x)
        trip_joint = self.trip_joint(x)

        # organic head
        organic_aspect = self.organic_aspect(x)
        organic_polarity = self.organic_polarity(x)
        organic_joint = self.organic_joint(x)

        # SemEval14res head
        SemEval14res_aspect = self.SemEval14res_aspect(x)
        SemEval14res_polarity = self.SemEval14res_polarity(x)
        SemEval14res_joint = self.SemEval14res_joint(x)

        # SemEval14lap head
        SemEval14lap_aspect = self.SemEval14lap_aspect(x)
        SemEval14lap_polarity = self.SemEval14lap_polarity(x)
        SemEval14lap_joint = self.SemEval14lap_joint(x)
        
        # SemEval16res head
        SemEval16res_aspect = self.SemEval16res_aspect(x)
        SemEval16res_polarity = self.SemEval16res_polarity(x)
        SemEval16res_joint = self.SemEval16res_joint(x)
 
        # SemEval16lap head
        SemEval16lap_aspect = self.SemEval16lap_aspect(x)
        SemEval16lap_polarity = self.SemEval16lap_polarity(x)
        SemEval16lap_joint = self.SemEval16lap_joint(x)

        return {
                   'aspect': trip_aspect, 'polarity': trip_polarity, 'joint': trip_joint
               }, {
                   'aspect': organic_aspect, 'polarity': organic_polarity, 'joint': organic_joint
               }, {
                   'aspect': SemEval14res_aspect, 'polarity': SemEval14res_polarity, 'joint': SemEval14res_joint
               }, {
                   'aspect': SemEval14lap_aspect, 'polarity': SemEval14lap_polarity, 'joint': SemEval14lap_joint
               }, {
                   'aspect': SemEval16res_aspect, 'polarity': SemEval16res_polarity, 'joint': SemEval16res_joint
               }, {
                   'aspect': SemEval16lap_aspect, 'polarity': SemEval16lap_polarity, 'joint': SemEval16lap_joint
               }


class MultiheadConfig(nn.Module):
    def __init__(self, config, embedding_dim=768, drop_prob=0.5):
        super(MultiheadConfig, self).__init__()

        self.backbone_dims = [embedding_dim // 2, embedding_dim // 4]

        # backbone
        self.backbone = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=self.backbone_dims[0]),
            nn.BatchNorm1d(self.backbone_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=self.backbone_dims[0], out_features=self.backbone_dims[1]),
            nn.BatchNorm1d(self.backbone_dims[1]),
            nn.ReLU()
        )

        # no sigmoid -> returns logits for torch.nn.BCEWithLogitsLoss()
        self.heads = {}
        for domain in config:
            self.heads[domain["name"]] = {}
            for head in domain["heads"]:
                if domain["heads"][head]:
                    self.heads[domain["name"]][head] = nn.Linear(in_features=self.backbone_dims[1], out_features=domain["classes"][head])

    def forward(self, input):
        x = self.backbone(input)

        head_outputs = {}
        for domain in self.heads:
            head_outputs[domain] = {}
            for head in self.heads[domain]:
                head_outputs[domain][head] = self.heads[domain][head](x)

        return head_outputs
