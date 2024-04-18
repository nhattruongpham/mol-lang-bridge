# from lang.t5 import T5ForMultimodalConditionalGeneration
# from vision.swin import SwinTransformer
# from transformers import AutoTokenizer
# import torch


# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("QizhiPei/biot5-base")
    
#     model = T5ForMultimodalConditionalGeneration.from_pretrained(
#         "QizhiPei/biot5-base",
#         n_attention_heads=8,
#         use_forget_gate=False,
#         visual_feature_dim=1536,
#         text_feature_dim=768,
#         intermidate_dim=256 
#     )
    
#     sample_image = torch.randn(size=(1,3,224,224))
    
#     vision_model = SwinTransformer(
#         embed_dim=192,
#         depths=[2,2,18,2],
#         num_heads=[6,12,24,48],
#         num_classes=0
#     )
#     vision_model.load_state_dict(
#         torch.load('/Users/jaydentran1909/Documents/codes/research/mol-lang-bridge/src/weights/swin_transform_focalloss.pth',
#                    map_location=torch.device('cpu'))['encoder']
#     )
    
#     image_features = vision_model.forward_features(sample_image, avgpool=False)
#     # print(image_features.shape)
    
#     input = tokenizer("[C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][=Branch1][C][=O][O][C@H1][Branch2][Ring1][#Branch2][C][O][C][=Branch1][C][=O][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][O][P][=Branch1][C][=O][Branch1][C][O][O][C][C@@H1][Branch1][C][O][C][O][P][=Branch1][C][=O][Branch1][C][O][O][C][C@@H1][Branch2][Ring1][Branch1][C][O][C][=Branch1][C][=O][C][C][C][C][C][C][C][C][C][Branch1][C][C][C][C][O][C][=Branch1][C][=O][C][C][C][C][C][C][C][C][C][C][C][C][Branch1][C][C][C]", 
#                       add_special_tokens=True,
#                       max_length=512,
#                       padding='max_length',
#                       truncation=True,
#                       return_attention_mask=True,
#                       return_tensors='pt')
#     output = tokenizer('The molecule is a stabilizing mitochondrial structure, proton trap for oxidative phosphorylation, cholesterol translocation that impacts barth syndrome and tangier disease. The molecule is a stabilizing cytochrome oxidase and a apoptosis that impacts aging, diabetic heart disease, and non-alcoholic fatty liver disease.', 
#                        add_special_tokens=True,
#                        max_length=512,
#                        padding='max_length',
#                        truncation=True,
#                        return_attention_mask=True,
#                        return_tensors='pt')
    
#     input_ids = input['input_ids']
#     attention_mask = input['attention_mask']
#     labels = output['input_ids']
    
#     decoder_input_ids = labels[:, :-1].contiguous()
#     decoder_target_ids = labels[:, 1:].clone().detach()
#     decoder_target_ids[labels[:, 1:] == tokenizer.pad_token_id] = -100
#     # image_features = torch.randn(size=(1, 768, 1536))
#     # print(text)
    
#     output = model(input_ids=input_ids, 
#                         attention_mask=attention_mask, 
#                         decoder_input_ids=decoder_input_ids,
#                         labels=decoder_target_ids,
#                         image_features=image_features.float())