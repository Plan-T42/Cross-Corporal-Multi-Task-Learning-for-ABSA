import torch.nn as nn


class MultiheadV2(nn.Module):
    def __init__(self, aspect_classes, polarity_classes, joint_classes, embedding_dim=768, drop_prob=0.2):
        super(MultiheadV2, self).__init__()

        self.backbone_dims = [embedding_dim // 4, embedding_dim // 4]

        # self.backbone = nn.Sequential(
        #     nn.LSTM(input_size=embedding_dim, hidden_size=self.backbone_dims[0], num_layers=1, 
        #             dropout=drop_prob, batch_first=True, bidirectional=True),
        #     nn.Linear(in_features=2 * self.backbone_dims[0], out_features=self.backbone_dims[1]),
        # )      

        self.input_size = embedding_dim
        self.dropout = drop_prob
        self.lstm = nn.LSTM(self.input_size, hidden_size=self.backbone_dims[0], num_layers=1, 
                            dropout=self.dropout, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(in_features=2 * self.backbone_dims[0], out_features=self.backbone_dims[1])

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
        # x = self.backbone(input)
        x, (h_n, h_c) = self.lstm(input)
        x = x[-1, :, :]
        x = self.classifier(x)

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
    def __init__(self, config, embedding_dim=768, drop_prob=0.2):
        super(MultiheadConfig, self).__init__()
        
        # tuning self.backbone_dims[0](embdding_dim//2 or embdding_dim) while training the model
        self.backbone_dims = [embedding_dim, embedding_dim // 4]
        self.input_size = embedding_dim
        self.dropout = drop_prob

        # tuning self.num_layers (basic or stacked LSTM) while training the model
        self.lstm = nn.LSTM(self.input_size, hidden_size=self.backbone_dims[0], num_layers=1, 
                            dropout=self.dropout, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(in_features=2 * self.backbone_dims[0], out_features=self.backbone_dims[1])

        # no sigmoid -> returns logits for torch.nn.BCEWithLogitsLoss()
        #self.heads = {}
        config_dict = {}        
        for domain in config:
            #self.heads[domain["name"]] = {}
            domain_dict = {}
            for head in domain["heads"]:
                if domain["heads"][head]:
                    #self.heads[domain["name"]][head] = nn.Linear(in_features=self.backbone_dims[1], out_features=domain["classes"][head])
                    domain_dict[head] = nn.Linear(in_features=self.backbone_dims[1], out_features=domain["classes"][head])
            config_dict[domain["name"]] = nn.ModuleDict(domain_dict)

        self.heads = nn.ModuleDict(config_dict)

    def forward(self, input):
    #     x = self.backbone(input)
        x, (h_n, h_c) = self.lstm(input)
        x = x[-1, :, :]
        x = self.classifier(x)

        head_outputs = {}
        for domain in self.heads:
            head_outputs[domain] = {}
            for head in self.heads[domain]:
                head_outputs[domain][head] = self.heads[domain][head](x)

        return head_outputs

        
