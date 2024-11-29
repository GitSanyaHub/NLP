import torch
from torch.nn import functional as F
from transformers import Trainer

# https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=1.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    # Переобпределение функции расчета ошибки
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Получаем истинные griund truth метки
        labels = inputs.pop("labels")
        
        # Получаем логиты студента
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Получаем логиты учителя
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Расчет KL дивергенции
        distill_loss = self._compute_distillation_loss(student_logits, teacher_logits)

        # Ошибка на истиных метках
        hard_loss = torch.nn.CrossEntropyLoss()(student_logits.view(-1, self.model.config.num_labels), labels.view(-1))

        # Общая ошибка
        loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

        return (loss, outputs) if return_outputs else loss

    def _compute_distillation_loss(self, student_logits, teacher_logits):
        # Вероятности студента и учителя
        student_probs = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Расчет расхожения распределения вероятностей
        return torch.nn.functional.kl_div(student_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)


