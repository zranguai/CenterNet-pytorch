import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from models.loss import focal_loss, reg_l1_loss


def fit_one_epoch(model_train, model, optimizer, epoch, epoch_step, epoch_step_val, train_dataloader, val_dataloader, end_epoch, cuda):
    total_r_loss = 0  # 回归损失
    total_c_loss = 0  # 分类损失
    total_loss = 0  # 总损失
    val_loss = 0  # 验证集损失

    model_train.train()
    print('start epoch train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataloader):
            if iteration >= epoch_step:  # 迭代数超出数据量退出
                break
            # 将ndarray的数据类型转换成tensor格式的
            with torch.no_grad():
                if cuda:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch]
                else:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in batch]
            # 取出数据
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            # 计算loss
            hm, wh, offset = model_train(batch_images)  # predict

            c_loss = focal_loss(hm, batch_hms)
            wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
            loss = c_loss + wh_loss + off_loss

            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_r_loss += wh_loss.item() + off_loss.item()

            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'total_r_loss': total_r_loss / (iteration + 1),
                                'total_c_loss': total_c_loss / (iteration + 1),
                                })
            pbar.update(1)

    print('End epoch train')
    writer = SummaryWriter('loss_logs1')  # 进行损失可视化
    # 验证数据集  --> 将可视化写到这里
    model_train.eval()
    print("Start val")
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_dataloader):
            if iteration >= epoch_step:  # 迭代数超出数据量退出
                break
            # 将ndarray的数据类型转换成tensor格式的
            with torch.no_grad():
                if cuda:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch]
                else:
                    batch = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in batch]
                # 取出数据
                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

                # 计算loss
                hm, wh, offset = model_train(batch_images)  # predict

                c_loss = focal_loss(hm, batch_hms)
                wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                loss = c_loss + wh_loss + off_loss
                val_loss += loss.item()

                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    writer.add_scalar("total_loss", total_loss, epoch)
    writer.add_scalar("total_c_loss", total_c_loss, epoch)
    writer.add_scalar("total_r_loss", total_r_loss, epoch)
    writer.close()  # 可视化关闭
    print("Finish Validation")

    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

    if epoch >= 9:  # 进行保存模型
        torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
