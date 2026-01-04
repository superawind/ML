import logging
import os 
from typing import Optional

import torch
from transformers.trainer import Trainer
from modeling import CrossEncoder

logger = logging.getLogger(__name__)

class CETrainer(Trainer):
    def compute_loss(self, model: CrossEncoder, inputs):
        return model(inputs)['loss']
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("save model checkpoint to %s", output_dir)

        if not hasattr(self.model, 'save_pretrained'):
             raise NotImplementedError(f'MODEL {self.model.__class__.__name__} ' f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if  self.tokenizer and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))