library(ggplot2)

setwd("/home/u/Documents/university/machine_learning/BeautyFlower/statistics")

## DISCRIMINATORS ##
# Load the models
dis_loss2 = read.csv("model2/disLoss.csv", header = FALSE)
names(dis_loss2) = c("time", "epoch", "loss")
dis_loss2$blocks = "4"
dis_loss3 = read.csv("model3/disLoss.csv", header = FALSE)
names(dis_loss3) = c("time", "epoch", "loss")
dis_loss3$blocks = "8"
dis_loss4 = read.csv("model4/disLoss.csv", header = FALSE)
names(dis_loss4) = c("time", "epoch", "loss")
dis_loss4$blocks = "12"
dis_loss5 = read.csv("model5/disLoss.csv", header = FALSE)
names(dis_loss5) = c("time", "epoch", "loss")
dis_loss5$blocks = "16"

# Row bind
total_dis_loss = rbind(dis_loss2, dis_loss3, dis_loss4, dis_loss5)
total_dis_loss$blocks = as.factor(total_dis_loss$blocks)

# Make plot
ggplot(total_dis_loss , aes(epoch, loss , colour=blocks , group=blocks)) + geom_line() +
  labs(title="Discriminator loss after each epoch",
       x ="Epochs trained", y = "Loss value", color='Amount of \nresidual blocks')

## GENERATORS ##
gen_loss2 = read.csv("model2/genLoss.csv", header = FALSE)
names(gen_loss2) = c("time", "epoch", "loss")
gen_loss2$blocks = "4"
gen_loss3 = read.csv("model3/genLoss.csv", header = FALSE)
names(gen_loss3) = c("time", "epoch", "loss")
gen_loss3$blocks = "8"
gen_loss4 = read.csv("model4/genLoss.csv", header = FALSE)
names(gen_loss4) = c("time", "epoch", "loss")
gen_loss4$blocks = "12"
gen_loss5 = read.csv("model5/genLoss.csv", header = FALSE)
names(gen_loss5) = c("time", "epoch", "loss")
gen_loss5$blocks = "16"

# Row bind
total_gen_loss = rbind(gen_loss2, gen_loss3, gen_loss4, gen_loss5)
total_gen_loss$blocks = as.factor(total_gen_loss$blocks)

# Make plot
ggplot(total_gen_loss , aes(epoch, loss , colour=blocks , group=blocks)) + geom_line() +
  labs(title="Generator loss after each epoch",
       x ="Epochs trained", y = "Loss value", color='Amount of \nresidual blocks')

## PSNR VALUES ##
# Read values
psnr2 = read.csv("model2/psnr.csv", header = FALSE)
psnr2$epoch = psnr2$V1
psnr2$average = apply(psnr2[,2:21], 1, mean)
psnr2$blocks = "4"
psnr3 = read.csv("model3/psnr.csv", header = FALSE)
psnr3$epoch = psnr3$V1
psnr3$average = apply(psnr3[,2:21], 1, mean)
psnr3$blocks = "8"
psnr4 = read.csv("model4/psnr.csv", header = FALSE)
psnr4$epoch = psnr4$V1
psnr4$average = apply(psnr4[,2:21], 1, mean)
psnr4$blocks = "12"
psnr5 = read.csv("model5/psnr.csv", header = FALSE)
psnr5$epoch = psnr5$V1
psnr5$average = apply(psnr5[,2:21], 1, mean)
psnr5$blocks = "16"

# Stack data
total_psnr = rbind(psnr2, psnr3, psnr4, psnr5)
total_psnr$blocks = as.factor(total_psnr$blocks)

# Make plot
ggplot(total_psnr , aes(epoch, average , colour=blocks , group=blocks)) + geom_line() +
  labs(title="Average PSNR after each epoch",
       x ="Epochs trained", y = "PSNR value", color='Amount of \nresidual blocks')

## SSIM VALUES ##
# Read values
ssim2 = read.csv("model2/ssim.csv", header = FALSE)
ssim2$epoch = ssim2$V1
ssim2$average = apply(ssim2[,2:21], 1, mean)
ssim2$blocks = "4"
ssim3 = read.csv("model3/ssim.csv", header = FALSE)
ssim3$epoch = ssim3$V1
ssim3$average = apply(ssim3[,2:21], 1, mean)
ssim3$blocks = "8"
ssim4 = read.csv("model4/ssim.csv", header = FALSE)
ssim4$epoch = ssim4$V1
ssim4$average = apply(ssim4[,2:21], 1, mean)
ssim4$blocks = "12"
ssim5 = read.csv("model5/ssim.csv", header = FALSE)
ssim5$epoch = ssim5$V1
ssim5$average = apply(ssim5[,2:21], 1, mean)
ssim5$blocks = "16"

# Stack data
total_ssim = rbind(ssim2, ssim3, ssim4, ssim5)
total_ssim$blocks = as.factor(total_ssim$blocks)

# Make plot
ggplot(total_ssim , aes(epoch, average , colour=blocks , group=blocks)) + geom_line() +
  labs(title="Average SSIM after each epoch",
       x ="Epochs trained", y = "SSIM value", color='Amount of \nresidual blocks')
