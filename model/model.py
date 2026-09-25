from asyncio import transports
import pdb
import os
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from focal_loss.focal_loss import FocalLoss as FL

from run.utils import Show_Samples
from run.Args import args
from model.ResNet3D import generate_model as resnet3D
from model.MSLA import MSLA as msla

def ini_weights(module_list:list):
    for m in module_list:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Res_3D_Encoder(nn.Module):
    def __init__(self, kargs = args, **kwargs) -> None:
        super().__init__()
        layer = kargs.model_depth
        if layer == 50:
            self.model = resnet3D(kargs)
            self.feature_channel = 2048
        elif layer == 18:
            self.model = resnet3D(kargs)
            self.feature_channel = 512

    def forward(self, x, squeeze_to_vector = False, pool="max"):
        assert x.dim() == 5, 'Wrong input dimension'
        if pool == "avg":
            pool_func = F.adaptive_avg_pool3d
        else:
            pool_func = F.adaptive_max_pool3d
        x = self.model(x)
        if squeeze_to_vector:
            x = pool_func(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            x = x.transpose(1,2)
            x = pool_func(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x

class Res_2D_Encoder(nn.Module):
    def __init__(self, kargs, *args,**kwargs) -> None:
        raise Exception("This module is deprecated")
        super().__init__()
        self.kargs = kargs
        if kargs.model_depth == 50:
            self.model = models.resnet50(weights="IMAGENET1K_V2")
            self.feature_channel = 2048
        elif kargs.model_depth == 18:
            self.model = models.resnet18(weights="IMAGENET1K_V1")
            self.feature_channel = 512
        
    def forward(self, input, squeeze_to_vector = False, pool="avg"):
        # x.shape = (1, 1, slice, h, w)
        assert input.dim() == 5, 'Wrong input dimension'
        assert input.shape[0]==1 and input.shape[1] == 1, "only support batchsize=1, but got shape"+str(input.shape) 
        x = torch.cat((input, input, input), dim=1).transpose(1, 2).squeeze(0) # slice, c, h, w
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.unsqueeze(0) # 1, slice, c, h, w
        if squeeze_to_vector:
            x = x.transpose(1,2) #1, c, d, h, w
            x = F.adaptive_max_pool3d(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            #-> 1, slice, c, 1
            x = F.adaptive_max_pool3d(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x

class Eff_2D_Encoder(nn.Module):
    def __init__(self, kargs, *args, **kwargs) -> None:
        raise Exception("This module is deprecated")
        super().__init__(*args, **kwargs)
        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.feature_channel = 1280

    def forward(self, input, squeeze_to_vector = False, pool="avg"):
        # x.shape = (1, 1, slice, h, w)
        assert input.dim() == 5, 'Wrong input dimension'
        assert input.shape[0]==1 and input.shape[1] == 1, "only support batchsize=1, but got shape"+str(input.shape) 
        x = torch.cat((input, input, input), dim=1).transpose(1, 2).squeeze(0) # slice, c, h, w
        x = self.model.features(x)
        s, c, h, w = x.shape
        x = x.unsqueeze(0)
        if squeeze_to_vector:
            #->b, c, 1
            x = x.transpose(1,2)
            x = F.adaptive_max_pool3d(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            #-> b, slice, c
            x = F.adaptive_max_pool3d(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x
        
class MSEA(nn.Module):
    def __init__(self, embed_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb_dim = embed_dim
        self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        ini_weights(self.modules())

    def forward(self, main_f, co_f1, co_f2):
        # (batch, channel, d)
        res = main_f
        q = self.mq(main_f)
        k1 = self.mk1(co_f1).permute(0, 2, 1)
        k2 = self.mk2(co_f2).permute(0, 2, 1)
        v1 = self.mv1(co_f1)
        v2 = self.mv2(co_f2)
        att1 = torch.matmul(q, k1)/np.sqrt(self.emb_dim)
        att1 = torch.softmax(att1, -1)
        att2 = torch.matmul(q, k2)/np.sqrt(self.emb_dim)
        att2 = torch.softmax(att2, -1)
        out1 = torch.matmul(att1, v1)
        out2 = torch.matmul(att2, v2)
        self.attmap1 = att1.detach().cpu()
        self.attmap2 = att2.detach().cpu()
        f = self.norm(0.5*(out1+out2)+res)
        f = f.transpose(1, 2) # b, d, c
        f = F.adaptive_max_pool1d(f, 1) # b, d, 1
        f = torch.flatten(f, start_dim=1) # b, d
        return f

class MSLA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.msla = msla(
                        mlp_dim=3072,
                        num_layers=6,
                        num_heads=8,
                        hidden_size=512,
                        modality_fusion=('modal0', 'modal1', 'modal2'),
                        fusion_layer=2,
                        n_bottlenecks=6,
                        classifier='token',
                    )

    def forward(self, main_f, co_f1, co_f2):
        # (batch, channel, d)
        f = self.msla(main_f, co_f1, co_f2)
        return f

class MMA(nn.Module):
    def __init__(self, feature_channel, kargs = args, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_channel = feature_channel
        self.reportlinear = nn.Linear(768, feature_channel*4)
        self.promptlinear = nn.Linear(768, feature_channel*4)
        self.mq = nn.Linear(feature_channel*4, feature_channel*4, bias=False)
        self.mk1 = nn.Linear(feature_channel*4, feature_channel*4, bias=False)
        self.mk2 = nn.Linear(feature_channel*4, feature_channel*4, bias=False)
        self.mv1 = nn.Linear(feature_channel*4, feature_channel*4, bias=False)
        self.mv2 = nn.Linear(feature_channel*4, feature_channel*4, bias=False)
        self.norm = nn.LayerNorm(feature_channel*4)
        ini_weights(self.modules())

    def forward(self, main_f, co_f1, co_f2):
        # (batch, channel, d)
        co_f1 = self.reportlinear(co_f1)
        co_f2 = self.promptlinear(co_f2)
        main_f = main_f.unsqueeze(1)

        res = main_f
        q = self.mq(main_f)
        k1 = self.mk1(co_f1).permute(0, 2, 1)
        k2 = self.mk2(co_f2).permute(0, 2, 1)
        v1 = self.mv1(co_f1)
        v2 = self.mv2(co_f2)
        att1 = torch.matmul(q, k1)/np.sqrt(self.feature_channel*4)
        att1 = torch.softmax(att1, -1)
        att2 = torch.matmul(q, k2)/np.sqrt(self.feature_channel*4)
        att2 = torch.softmax(att2, -1)
        out1 = torch.matmul(att1, v1)
        out2 = torch.matmul(att2, v2)
        self.attmap1 = att1.detach().cpu()
        self.attmap2 = att2.detach().cpu()
        # f = self.norm(out2+res)
        f = self.norm(out1+out2+res)
        f = f.transpose(1, 2) # b, d, c
        f = F.adaptive_max_pool1d(f, 1) # b, d, 1
        f = torch.flatten(f, start_dim=1) # b, d
        return f
    
class Branch_Classifier(nn.Module):
    def __init__(self, classnum, feature_channel, dropout_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(feature_channel, classnum))
        ini_weights(self.classifiers)

    def forward(self, f):
        if f.dim() == 3:
            f = f.squeeze(2)
        return self.classifiers(f)

class Multi_view_Knee(nn.Module):
    def __init__(self, backbone = 'ResNet', encoder_layer = 18, pretrain = True, parallel_device = True, kargs = args) -> None:
        super().__init__()
        self.kargs = kargs
        self.class_num = kargs.ClassNum
        self.para_device = parallel_device
        self.branch = kargs.active_branch
        # set parallel devices
        if torch.cuda.device_count() == 2 and self.para_device:
            self.device_list = ["cuda:%d"%x for x in range(torch.cuda.device_count())]
        else:
            self.device_list = ["cuda:0"]*3
        self.dropout_rate = 0.05
        self.backbone = kargs.backbone
        self.module_list = []
        self.__make_encoder__(self.backbone)
        self.__make_MSEA__()
        self.__make_MSLA__()
        self.__make_MMA__()
        self.__make_classifier__()
        # self.mining_conv = nn.Conv3d(1, 12, (12,12,12))
        self.mining_conv = nn.Conv2d(1, 2, (2,2))
        ini_weights([self.mining_conv, self.multi_view_classifier])
        model_param = sum(np.prod(v.size()) for name, v in self.named_parameters()) / 1e6
        print('model param = %f MB'%model_param)

        if sum(self.branch) == 3:
            try:
                self.save_or_load_encoder_para("load", kargs.pretrain_folder)
                print('loading pretrained success')
            except:
                print('loading pretrained failed, using random init')

    def __make_encoder__(self, backbone_name):
        if backbone_name == 'ResNet':
            encoder_func = Res_2D_Encoder
        elif backbone_name == 'ResNet3D':
            encoder_func = Res_3D_Encoder
        elif backbone_name == "EffNet":
            encoder_func = Eff_2D_Encoder
        else:
            raise Exception('Wrong Backbone Name!')
        self.encoder_func = encoder_func
        if self.branch[0]:
            self.sag_PDW_spair_enc = encoder_func(self.kargs).to(self.device_list[0])
            self.sag_T1w_tse_enc = encoder_func(self.kargs).to(self.device_list[0])
            self.sag_PDW_atse_enc = encoder_func(self.kargs).to(self.device_list[0])
            # if not self.kargs.no_cross_modal:
            #     self.t2w_enc = encoder_func(self.kargs).to(self.device_list[0])
        if self.branch[1]:
            self.sag_enc = encoder_func(self.kargs).to(self.device_list[1])
            self.cor_enc = encoder_func(self.kargs).to(self.device_list[1])
            self.axi_enc = encoder_func(self.kargs).to(self.device_list[1])
            # if not self.kargs.no_cross_modal:
            #     self.t1w_enc = encoder_func(self.kargs).to(self.device_list[0])
        return

    def __make_MSEA__(self):
        if self.kargs.no_co_att:
            return
        emb_dim = self.kargs.emb_dim
        if self.branch[0]:
            self.mmb_att = MSEA(emb_dim).to(self.device_list[0])
        if self.branch[1]:
            self.mvb_att = MSEA(emb_dim).to(self.device_list[1])
        return
    
    def __make_MSLA__(self):
        if self.kargs.no_mbt:
            return
        if self.branch[0]:
            self.mmb_mbt = MSLA().to(self.device_list[0])
        if self.branch[1]:
            self.mvb_mbt = MSLA().to(self.device_list[1])
        return
    
    def __make_MMA__(self):
        if self.kargs.no_cross_modal:
            return
        if self.branch[0]:
            self.mmb_cross_att = MMA(self.sag_enc.feature_channel, self.kargs).to(self.device_list[0])
        if self.branch[1]:
            self.mvb_cross_att = MMA(self.cor_enc.feature_channel, self.kargs).to(self.device_list[1])
        return
    
    def __make_classifier__(self):
        if self.branch[0]:
            if self.kargs.no_mbt:
                self.mmb_classifier = Branch_Classifier(self.class_num, self.sag_PDW_spair_enc.feature_channel, self.dropout_rate).to(self.device_list[0])
            else:
                self.mmb_classifier = Branch_Classifier(self.class_num, self.sag_PDW_spair_enc.feature_channel*4+2, self.dropout_rate).to(self.device_list[0])
        if self.branch[1]:
            if self.kargs.no_mbt:
                self.mvb_classifier = Branch_Classifier(self.class_num, self.sag_PDW_spair_enc.feature_channel, self.dropout_rate).to(self.device_list[0])
            else:
                self.mvb_classifier = Branch_Classifier(self.class_num, self.sag_PDW_spair_enc.feature_channel*4+2, self.dropout_rate).to(self.device_list[1])
    
        if self.kargs.no_corr_mining:
            if self.kargs.no_mbt:
                self.multi_view_classifier = Branch_Classifier(self.class_num, self.sag_PDW_spair_enc.feature_channel*2, self.dropout_rate).to(self.device_list[0])
            else:
                self.multi_view_classifier = Branch_Classifier(self.class_num, self.sag_PDW_spair_enc.feature_channel*8+4, self.dropout_rate).to(self.device_list[0])

        else:
            self.multi_view_classifier = nn.Sequential(nn.Linear(2, 2)).to(self.device_list[0])
        return

    def __mm_branch__(self, input, clinic, report, prompt, device = None):
        sag_PDW_spair_img, cor_image, axi_image, sag_T1w_tse_img, sag_PDW_atse_img = input

        if self.kargs.no_co_att:
            pdw_f = self.sag_PDW_spair_enc(sag_PDW_spair_img, squeeze_to_vector = True)
        else:
            if self.kargs.show_patch_sample:
                for i in range(0, self.kargs.INPUT_DIM, 10):
                    Show_Samples(sag_PDW_spair_img[0][0][i], title="sag_img_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(sag_T1w_tse_img[0][0][i], title="sag_cor_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(sag_PDW_atse_img[0][0][i], title="sag_axi_%d"%i, bcdhw=True, save_path="./Pic")        
            if device != None:
                sag_PDW_spair_img = sag_PDW_spair_img.to(device, non_blocking=True)
                sag_T1w_tse_img = sag_T1w_tse_img.to(device, non_blocking=True)
                sag_PDW_atse_img = sag_PDW_atse_img.to(device, non_blocking=True)

            main_f = self.sag_PDW_spair_enc(sag_PDW_spair_img)
            co_f1 = self.sag_T1w_tse_enc(sag_T1w_tse_img)
            co_f2 = self.sag_PDW_atse_enc(sag_PDW_atse_img)
            pdw_f = self.mmb_att(main_f, co_f1, co_f2)

            if self.kargs.no_mbt: 
                pdw_f = pdw_f
            else:
                pdw_f_mbt = self.mmb_mbt(main_f, co_f1, co_f2)
                pdw_f = torch.cat((pdw_f, pdw_f_mbt), dim=1)

        if self.kargs.no_cross_modal:
            pred = self.mmb_classifier(pdw_f)
        else:
            pdw_f = self.mmb_cross_att(pdw_f, report, prompt)
            pdw_f = torch.cat((pdw_f, clinic), dim=1)
            pred = self.mmb_classifier(pdw_f)
        return pred, pdw_f
    
    def __mv_branch__(self, input, clinic, report, prompt, device = None):
        sag_PDW_spair_img, cor_image, axi_image, sag_T1w_tse_img, sag_PDW_atse_img = input #(b,1,244,244,244)
        
        if self.kargs.no_co_att:
            pdw_f = self.sag_PDW_spair_enc(sag_PDW_spair_img, squeeze_to_vector = True)
        else:
            # cor_image = cor_image.transpose(2,4)
            # axi_image = torch.rot90(axi_image.transpose(2,4), k=1, dims=[3,4]).detach()
            # Show sample
            if self.kargs.show_patch_sample:
                for i in range(0, self.kargs.INPUT_DIM, 10):
                    Show_Samples(sag_PDW_spair_img[0][0][i], title="cor_img_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(cor_image[0][0][i], title="cor_sag_%d"%i, bcdhw=True, save_path="./Pic")
                    Show_Samples(axi_image[0][0][i], title="cor_axi_%d"%i, bcdhw=True, save_path="./Pic")
            if device != None:
                sag_PDW_spair_img = sag_PDW_spair_img.to(device, non_blocking=True)
                cor_image = cor_image.to(device, non_blocking=True)
                axi_image = axi_image.to(device, non_blocking=True)

            main_f = self.sag_enc(sag_PDW_spair_img) #(b,28,512)
            co_f1 = self.cor_enc(cor_image)
            co_f2 = self.axi_enc(axi_image)
            pdw_f = self.mvb_att(main_f, co_f1, co_f2)

            if self.kargs.no_mbt: 
                pdw_f = pdw_f
            else:
                pdw_f_mbt = self.mvb_mbt(main_f, co_f1, co_f2)
                pdw_f = torch.cat((pdw_f, pdw_f_mbt), dim=1)

        if self.kargs.no_cross_modal:
            pred = self.mvb_classifier(pdw_f)
        else:
            pdw_f = self.mvb_cross_att(pdw_f, report, prompt)
            pdw_f = torch.cat((pdw_f, clinic), dim=1)
            pred = self.mvb_classifier(pdw_f)
        return pred, pdw_f     

    def __discovery__(self, mmb_pred, mm_pdwf, mvb_pred, mv_pdwf, clinic, report, prompt, device = None):
        with torch.no_grad():
            mmb_pred = torch.sigmoid(mmb_pred)
            mvb_pred = torch.sigmoid(mvb_pred)

        if self.kargs.separate_final:
            mmb_pred = mmb_pred.detach().clone()
            mvb_pred = mvb_pred.detach().clone()

        if self.kargs.no_corr_mining:
            # pred_matrix = torch.cat((mmb_pred, mvb_pred), dim=1)
            # return self.multi_view_classifier(pred_matrix)
            
            pdwf = torch.cat((mm_pdwf, mv_pdwf), dim=1)
            return self.multi_view_classifier(pdwf)
        
        if device != None:
            mmb_pred = mmb_pred.to(device)
            mvb_pred = mvb_pred.to(device)

        union_prob = mmb_pred*mvb_pred
        mmb_t = mmb_pred.unsqueeze(2)
        mvb_t = mvb_pred.unsqueeze(1)
        pred_matrix = (mmb_t * mvb_t).unsqueeze(1)
        fin_att = torch.flatten(self.mining_conv(pred_matrix), start_dim=1)
        fin_pred = union_prob*fin_att
        return fin_pred

    def save_or_load_encoder_para(self, mode = "save", path = ""):
        if mode == "save":
            act_func = self.__save_para_
        elif mode == "load":
            act_func = self.__load_para_
        else:
            raise Exception("wrong mode name, should be save or load")
        
        if self.branch[0]:
            act_func(self.sag_PDW_spair_enc, 'sag_PDW_spair_enc', path = path)
            act_func(self.sag_T1w_tse_enc, 'sag_T1w_tse_enc', path = path)
            act_func(self.sag_PDW_atse_enc, 'sag_PDW_atse_enc', path = path)
        if self.branch[1]:
            act_func(self.sag_enc, 'sag_enc', path = path)
            act_func(self.cor_enc, 'cor_enc', path = path)
            act_func(self.axi_enc, 'axi_enc', path = path)

    def att_map(self, input):
        if self.kargs.no_co_att:
            print("no co-plane attention enabled")
        attmap_dict = {}
        if self.branch[0]:
            attmap_dict["mmb"] = [self.mmb_att.attmap1, self.mmb_att.attmap2]
        if self.branch[1]:
            attmap_dict["mvb"] = [self.mvb_att.attmap1, self.mvb_att.attmap2]

    def __save_para_(self, model, name, path=""):
        if path != "":
            save_path = path
        else:
            print("save path not specified, not saving")
            return
        try:
            torch.save(model.state_dict(), os.path.join(save_path, "%s_%s_para.pkl"%(model.__class__.__name__,name)))
        except:
            print("save para failed")

    def __load_para_(self, model, name, path=""):
        if path == "":
            save_path = self.kargs.pretrain_folder
        else:
            save_path = path
        model.load_state_dict(torch.load(os.path.join(save_path, "%s_%s_para.pkl"%(model.__class__.__name__,name))))

    def forward(self, input, clinic, report, prompt):
        # input: [[bz, channel, slice, h, w], []..]
        if self.branch[0]:
            mm_pred, mm_pdwf = self.__mm_branch__(input, clinic, report, prompt)
            final_pred = mm_pred
        if self.branch[1]:
            mv_pred, mv_pdwf = self.__mv_branch__(input, clinic, report, prompt)
            final_pred = mv_pred
        if sum(self.branch) == 1:
            return final_pred, final_pred, final_pred
        if sum(self.branch) == 2:
            final_pred = self.__discovery__(mm_pred, mm_pdwf, mv_pred, mv_pdwf, clinic, report, prompt)
            return final_pred, mm_pred, mv_pred

    def criterion(self, pred, label, act_task = -1, final = True):
        task_weights = torch.ones((2))
        task_weights = task_weights.tolist()

        pos_weights = torch.tensor(self.kargs.pos_weights).cuda()

        final_pred, mm_pred, mv_pred = pred
        mm_loss = 0.0
        mv_loss = 0.0
        final_loss = 0.0
        lossfunc = F.cross_entropy 
        lossfunc_final = Focal_Loss_with_logits

        for i in range(2):
            if i == act_task or act_task == -1:
                task_weight = task_weights[i]                
            else:
                task_weight = 0.01
            pos_wei = pos_weights[i * 4:(i + 1) * 4]
            subject_label = label[:, i]

            if self.branch[0]:
                mm_loss += task_weight* lossfunc(mm_pred[:, i * 4:(i + 1) * 4], subject_label, weight=pos_wei)
            if self.branch[1]:
                mv_loss += task_weight* lossfunc(mv_pred[:, i * 4:(i + 1) * 4], subject_label, weight=pos_wei)
            if final:
                final_loss += task_weight* lossfunc_final(final_pred[:, i * 4:(i + 1) * 4], subject_label, pos_weight=pos_wei)

        if sum(self.branch) == 1:
            loss = [mm_loss, mv_loss][self.branch.index(1)]
        elif final:
            loss= self.kargs.alpha*(mm_loss+mv_loss) + final_loss 
        else:
            loss = mm_loss+mv_loss

        return loss, loss.item()

class Pretrain_Encoder(nn.Module):
    def __init__(self, backbone = 'ResNet', encoder_layer = 18, pretrain = True, parallel_device = '1', kargs = args) -> None:
        super().__init__()
        self.kargs = kargs
        self.encoder = Res_3D_Encoder(kargs)
        self.classifier = Branch_Classifier(12, self.encoder.feature_channel, 0.05)
        plane = "sag"
        self.encoder_name = "%s_enc"%plane
        self.classifier_name = "%s_cls"%plane

    def forward(self, input):
        # input: [[bz, slice, channel, h, w], []..]
        sag_img, cor_img, axi_img, t2_img, t1_img = input
        x = sag_img
        f = self.encoder(x, squeeze_to_vector = True, pool="max")
        pred = self.classifier(f)
        return [pred]*4

    def save_or_load_encoder_para(self, mode = "save", path = ""):

        if mode == "load":
            self.load_state_dict(torch.load(os.path.join(path, "%s_%s_para.pkl"%(self.__class__.__name__,self.encoder_name))))
        elif mode == "save":
            if path != "":
                save_path = path
            else:
                print("save path not specified, not saving")
                return
            try:
                torch.save(self.encoder.state_dict(), os.path.join(save_path, "%s_%s_para.pkl"%(self.encoder.__class__.__name__,self.encoder_name)))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, "%s_%s_para.pkl"%(self.classifier.__class__.__name__,self.classifier_name)))
            except:
                print("save para failed")


    def criterion(self, pred, label, act_task = -1, final = False):
        lossfunc = F.binary_cross_entropy_with_logits
        final_pred, sag_pred, cor_pred, axi_pred = pred
        loss = lossfunc(final_pred, label)
        # pos_weights = torch.tensor(self.kargs.pos_weights).cuda()
        # for i in range(12):
        #     if i==act_task or act_task==-1:
        #         weight = 1.0
        #     else:
        #         weight = 0.1
        #     loss = loss + weight*lossfunc(final_pred[:,i:i+1], label[:,i:i+1], pos_weight = pos_weights[i])

        return loss, loss.item()

def Focal_Loss_with_logits( pred, label, pos_weight = None, gamma=2, reduction='mean'):
    pred = F.softmax(pred, dim=1)
    label = label.long()
    if pos_weight is not None:
        weights = []
        for i in range(4):
            positive_weight = pos_weight[i] / (1 + pos_weight[i])
            weights.append(torch.tensor([positive_weight]))
        weight = torch.tensor(weights)#[4,]
    else:
        weight = None
    fl = FL(gamma=gamma, weights=weight, reduction="none",eps=5e-6)
    loss = fl(pred, label)
    if reduction == "mean":
        loss = loss.sum()/loss.shape[0]
    elif reduction == "sum":
        loss = loss.sum()
    return loss

