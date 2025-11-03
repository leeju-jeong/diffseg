

def extract_text_embeddings(self, class_names, prompts, average=True):
    text_features = []
    for class_name in class_names:
        texts = [p.format(class_name) for p in prompts]
        texts = self.tokenize(texts).to(self.device)

        # 템플릿별 임베딩 (T, 512)
        class_embeddings = self.model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        if average:
            # 1) 템플릿 축 평균 (512,)
            class_embeddings_avg = class_embeddings.mean(dim=0)

            # 2) 평균 벡터 L2 정규화
            class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()

            # 3) (T,512) 뒤에 (1,512) 평균 벡터를 이어붙여 (T+1,512)
            class_embeddings = torch.cat(
                [class_embeddings, class_embeddings_avg.unsqueeze(0)],
                dim=0
            )

        text_features.append(class_embeddings)

    # 모든 클래스를 쌓아 최종 shape: (T+1, C, 512)
    text_features = torch.stack(text_features, dim=1).to(self.device)
    return text_features