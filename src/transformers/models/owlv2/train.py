import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
from torchvision import datasets, transforms

from transformers import Owlv2Processor
from transformers import Owlv2ForObjectDetection
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
from transformers import Owlv2Config
import torch.nn.functional as F
import json
from lvis.eval import LVISEval
from PIL import Image


class EvaluationDataset(Dataset):
    def __init__(self, data_root, ann_path, prompt, is_train=False):
        transform = transforms.Compose([
                transforms.Resize((960, 960)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.data_root = data_root
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.lvis_cat, self.lvis_annotations = self.load_lvis_annotations(ann_path)
        self.prompt = prompt
        self.transform = transform
        self.d_text, self.d_bbox, self.d_image_id, self.d_category_id = self.org_data(data_root, self.lvis_cat, self.lvis_annotations, prompt=prompt)
        self.files = list(self.d_text.keys())
        self.Nimages = len(self.d_text.keys())
        self.is_train = is_train
        print('Number of images: {N}'.format(N=self.Nimages))

    
    def check_annotations(self, lvis_annotations):
        cat = []
        for annotation in tqdm(lvis_annotations):
            cat.append(annotation['category_id'])
        return cat


    def load_lvis_annotations(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data['categories'], data['annotations']


    def get_image_path(self, img_list, name):
        return img_list[img_list.index(str(name).zfill(12)+'.jpg' )]


    def get_bbox(self, bbox):
        bbox_x1y1x2y2 = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        return torch.tensor([bbox_x1y1x2y2])


    def get_texts(self, prompt, lvis_cat, annotation):
        return prompt + lvis_cat[annotation['category_id']-1 ]['synonyms'][0]

    def org_data(self, root, lvis_cat, lvis_annotations, prompt=''):
        d_text = {}
        d_bbox = {}
        d_image_id = {}
        d_category_id = {}
        for ann in tqdm(lvis_annotations):
            image_path = os.path.join(root, 'val2017', str(ann['image_id']).zfill(12)+'.jpg')
            if not os.path.exists(image_path):
                image_path = os.path.join(root, 'train2017', str(ann['image_id']).zfill(12)+'.jpg')
            if not image_path in d_bbox.keys():
                d_text[image_path] = []
                d_bbox[image_path] = []
                d_image_id[image_path] = []
                d_category_id[image_path] = []
            text = self.get_texts(prompt, lvis_cat, ann) 
            gt_boxes = self.get_bbox(ann['bbox'])
            d_text[image_path].append(text)
            d_bbox[image_path].append(gt_boxes)
            d_image_id[image_path].append(ann['image_id'])
            d_category_id[image_path].append(ann['category_id'])
        return d_text, d_bbox, d_image_id, d_category_id

    def __len__(self):
        return 180
        # return self.Nimages

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path).convert('RGB')
        target_sizes = torch.Tensor([image.size[::-1]])
        image = self.transform(image)
        if self.is_train:
            texts = self.d_text[image_path][0]
            inputs = self.processor(text=texts, images=None, return_tensors="pt")
            inputs['pixel_values'] = image
            return inputs
        else:
            gt_boxes = self.d_bbox[image_path]
            texts = self.d_text[image_path] 
            image_id = self.d_image_id[image_path] 
            category_id = self.d_category_id[image_path] 
            inputs = self.processor(text=list(np.unique(texts)), images=None, return_tensors="pt")
            inputs['pixel_values'] = image
            return inputs, gt_boxes, target_sizes, texts, image_id, category_id


class Owlv2DisTrainer:
    def __init__(self, training_ds, validation_ds, args):
        os.makedirs(args.root, exist_ok=True)
        folder = self.create_results_folder()
        self.args = args
        self.mse_criterion = nn.L1Loss()
        self.bce_criterion = nn.BCELoss()
        self.folder = folder
        self._set_config_file()
        self.device = args.device
        if args.dtype == 'float16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self._init_teacher()
        self._init_student()
        self.optimizer = optim.AdamW(self.student.owlv2.vision_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.epoches = args.epoches
        self.threshold = args.th
        self._init_results_files()
        self.second_stage_flag = False
        self.final_stage_flag = False
        

    def _set_config_file(self):
        config = Owlv2Config()
        config.vision_config.num_attention_heads = args.num_attention_heads
        config.vision_config.num_hidden_layers = args.num_hidden_layers
        config.vision_config.intermediate_size = args.intermediate_size
        config.vision_config.name = args.name
        config.vision_config.image_size = 960
        self._save_config_json()
        self.config = config
    
    def _save_config_json(self):
        conf_dict = vars(self.args)
        with open(self.folder + '/config.json', 'w') as f:
            json.dump(conf_dict, f)

    def _init_teacher(self):
        self.teacher = Owlv2ForObjectDetection.from_pretrained(self.args.teacher).to(self.device).eval()
        self.processor = Owlv2Processor.from_pretrained(self.args.teacher)

    def _init_student(self):
        self.student = Owlv2ForObjectDetection(config=self.config).to(self.device)
        if args.resume == 1:
            cp = torch.load(args.resume_path + '/model.pth').state_dict()
            self.student.load_state_dict(cp, strict=True)
        else:
            self.student.load_state_dict(self.teacher.state_dict(), strict=False)
        for param in self.student.parameters():
            param.requires_grad = False
        for param in self.student.owlv2.vision_model.parameters():
            param.requires_grad = True

    def _init_results_files(self):
        self.predictions_path = self.folder + '/prediction.json'
        self.ground_truth_path = args.validation_ann_path
        self.f = open(self.folder + '/best.csv', 'w')
        self.save_args_to_file()
        self.f.write('ep,score\n')
        self.f.flush()

    def create_results_folder(self):
        current_datetime = datetime.now()
        date_time_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"results/results_{date_time_str}"
        os.makedirs(folder_name)
        return folder_name

    def set_inputs_device(self, inputs):
        inputs['input_ids'] = inputs['input_ids'].to(self.device).squeeze(dim=1).detach()
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device).squeeze(dim=1).detach()
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device).squeeze(dim=1).detach()
        return inputs

    def set_validation_inputs_device(self, inputs):
        inputs['input_ids'] = inputs['input_ids'].to(self.device).squeeze(dim=0).detach()
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device).squeeze(dim=1).detach()
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device).squeeze(dim=0).detach()
        return inputs
    
    def save_prediction(self, predictions):
        with open(self.predictions_path, 'w') as json_file:
                json.dump(predictions, json_file)

    def save_args_to_file(self):
        for key, value in vars(args).items():
            self.f.write("{}: {}\n".format(key, value))
        self.f.flush()

    def get_teacher_outputs(self, inputs):
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            z_t = teacher_outputs['vision_model_output']['last_hidden_state'].detach()
            objectness_t = teacher_outputs['objectness_logits'].detach()
            objectness_probs_t = F.sigmoid(objectness_t).detach()
            logits_t = teacher_outputs['logits'].detach()
            probs_t = F.sigmoid(logits_t).detach()
            boxes_t = teacher_outputs['pred_boxes'].detach()
        return z_t, objectness_t, objectness_probs_t, logits_t, probs_t, boxes_t
    
    def get_student_outputs(self, inputs):
        student_outputs = self.student(**inputs)
        z_s = student_outputs['vision_model_output']['last_hidden_state']
        objectness_s = student_outputs['objectness_logits']
        logits_s = student_outputs['logits']
        boxes_s = student_outputs['pred_boxes']
        return z_s, objectness_s, logits_s, boxes_s
    
    def representation_loss(self, z_s, z_t):
        return self.mse_criterion(z_s, z_t) 
    
    def head_loss(self, s, t, prob):
        loss = self.mse_criterion(s, t)
        if self.last_stage_flag:
            mask = (prob > self.threshold).squeeze()
            pos_loss = self.mse_criterion(s[mask], t[mask])
            loss = pos_loss + self.args.alpha * loss
        return loss
    
    def loss_function(self, student_outputs, teacher_outputs):
        z_t, objectness_t, objectness_probs_t, logits_t, probs_t, boxes_t = teacher_outputs
        z_s, objectness_s, logits_s, boxes_s = student_outputs
        z_loss = self.representation_loss(z_s, z_t) 
        if self.second_stage_flag:
            objectness_loss = self.head_loss(objectness_s, objectness_t, objectness_probs_t)
            logits_loss = self.head_loss(logits_s, logits_t, probs_t)
            bbox_loss = self.head_loss(boxes_s, boxes_t, objectness_probs_t)
            loss = z_loss + self.args.w0*objectness_loss + self.args.w1*logits_loss + self.args.w2*bbox_loss
        else:
            loss = z_loss
        return loss
    
    def training_single_epoch(self):
        pbar = tqdm(self.training_ds)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        loss_list = []
        self.student.train()
        for ix, (inputs) in enumerate(pbar):
            self.optimizer.zero_grad()
            inputs = self.set_inputs_device(inputs)
            with torch.amp.autocast(device_type=self.device, dtype=self.dtype, enabled=True):
                teacher_outputs = self.get_teacher_outputs(inputs)
                student_outputs = self.get_student_outputs(inputs)
                loss = self.loss_function(student_outputs, teacher_outputs)
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1)
            scaler.step(self.optimizer)
            scaler.update()
            loss_list.append(loss.item())
            loss_mean = np.mean(loss_list)
            pbar.set_description(f'Loss: {loss_mean:.4f}')               

    def update_flags(self, ep):
        if ep>= self.args.second_stage:
            self.second_stage_flag = True
        if ep>= self.args.last_stage:
            self.last_stage_flag = True

    def training(self):
        best = 0.0
        for epoch in range(self.epoches):
            self.update_flags(epoch)
            self.training_single_epoch()
            predictions = self.eval_ds()
            self.save_prediction(predictions)
            AP = self.calc_AP()
            if AP >= best:
                best = AP
                torch.save(self.student, self.folder + '/model.pth')
            self.f.write(str(epoch) + ',' + str(AP) + '\n')
            self.f.flush()

    @torch.no_grad
    def eval_ds(self):
        pbar = tqdm(self.validation_ds)
        predictions = []     
        self.student.eval()
        for ix, (inputs, gt_boxes, target_sizes, texts, image_id, category_id) in enumerate(pbar):
            texts = [texts[i][0] for i in range(len(texts))]
            inputs = self.set_validation_inputs_device(inputs)
            outputs = self.student(**inputs)
            results = self.get_output_results(outputs, target_sizes)
            results = self.op_nms(results)
            predictions = self.save_pred_json(results, texts, image_id, category_id, predictions)
        return predictions

    def save_pred_json(self, results, texts, image_id, category_id, predictions):
        cat = np.unique(texts)
        cid = np.unique(category_id)
        labels = results[0]["labels"]
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        id = image_id[0].item()
        for label, box, score in zip(labels, boxes, scores):
            d = {} 
            bbox = box.squeeze().long().tolist()
            d['image_id'] = int(id)
            d['category_id'] = int(cid[label])
            d['bbox'] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            d['score'] = score.item()
            d['area'] = d['bbox'][2] * d['bbox'][3]
            predictions.append(d)
        return predictions

    def calc_AP(self):
        lvis_eval = LVISEval(self.ground_truth_path, self.predictions_path, iou_type='bbox')
        lvis_eval.run()
        AP = lvis_eval.results['AP50']
        return AP
    
    def get_output_results(self, outputs, target_sizes):
        outputs['target_pred_boxes'] = outputs['pred_boxes'] 
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes.squeeze(dim=1).cuda(), threshold=self.threshold)
        return results

    def op_nms(self, results, iou_threshold=0.3):
        labels = results[0]["labels"]
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        keep = nms(boxes, scores, iou_threshold)
        results[0]["boxes"] = results[0]["boxes"][keep]
        results[0]["scores"] = results[0]["scores"][keep]
        results[0]["labels"] = results[0]["labels"][keep]
        return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
    parser.add_argument('--batch_size', type=int, default=3, help='')
    parser.add_argument('--nW', type=int, default=3, help='')
    parser.add_argument('--teacher', type=str,default='google/owlv2-base-patch16-ensemble', help='')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='')
    parser.add_argument('--intermediate_size', default=3072, type=int, help='')
    parser.add_argument('--name', default='convnext_tiny', type=str, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='')
    parser.add_argument('--epoches', default=100, type=int, help='')
    parser.add_argument('--th', default=0.1, type=float, help='')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='')
    parser.add_argument('--w0', default=0.1, type=float, help='')
    parser.add_argument('--w1', default=0.1, type=float, help='')
    parser.add_argument('--w2', default=1, type=float, help='')
    parser.add_argument('--alpha', default=0.025, type=float, help='')
    parser.add_argument('--resume', default=0, type=int, help='')
    parser.add_argument('--resume_path', default='results/results_2024-01-01_16-23-00', type=str, help='')
    parser.add_argument('--root', default='results', type=str, help='')
    parser.add_argument('--data_root', default='/home/talshah/PycharmProjects/owlv2/LVIS/', type=str, help='')
    parser.add_argument('--training_ann_path', default='/home/talshah/PycharmProjects/owlv2/LVIS/lvis_v1_train.json', type=str, help='')
    parser.add_argument('--validation_ann_path', default='/home/talshah/PycharmProjects/owlv2/LVIS/lvis_v1_val.json', type=str, help='')
    parser.add_argument('--dtype', default='float16', type=str, help='')
    parser.add_argument('--device', default='cuda', type=str, help='')
    parser.add_argument('--second_stage', default=4, type=int, help='')
    parser.add_argument('--last_stage', default=6, type=int, help='')
    args = parser.parse_args()
    
    training_dataset = EvaluationDataset(data_root= args.data_root, ann_path=args.training_ann_path, prompt='', is_train=True)
    validation_dataset = EvaluationDataset(data_root= args.data_root, ann_path=args.validation_ann_path, prompt='', is_train=False)
    ds = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nW, drop_last=True)
    ds_val = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    trainer = Owlv2DisTrainer(ds, ds_val, args)
    trainer.training()