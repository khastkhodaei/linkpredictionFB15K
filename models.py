import torch
import torch.nn as nn


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # ماتریس نشانه‌ها برای موجودیت‌ها و روابط
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # مقداردهی اولیه‌ی نشانه‌ها با توزیع نرمال
        nn.init.normal_(self.entity_embeddings.weight, std=0.01)
        nn.init.normal_(self.relation_embeddings.weight, std=0.01)

    def forward(self, head, relation, tail):
        # دریافت نشانه‌های موجودیت‌ها و روابط
        head_embedding = self.entity_embeddings(head)
        relation_embedding = self.relation_embeddings(relation)
        tail_embedding = self.entity_embeddings(tail)

        # محاسبه فاصله‌ی لیک
        distance = torch.norm(head_embedding + relation_embedding - tail_embedding, p=2, dim=-1)
        return distance
