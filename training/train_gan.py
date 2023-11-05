from utils.trainer import Trainer

def main():
    trainer = Trainer(dataset="mnist", mode="gan")
    # trainer.correct_module()
    trainer.train()

if __name__ == '__main__':
    main()