library(ggplot2)

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