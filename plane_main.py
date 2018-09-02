
###############################################
#  by Wang Siman
#  2018.6.18
#  There are three optional game modes:
#      1. Real player
#      2. Rule-based AI
#      3. Deep reinforcement learning AI
###############################################
import os
from PlaneGame import *
from plane_sprite import *
from sys import exit


class Game(object):
    def __init__(self):

        # 1.创建游戏窗口
        # Create initial windows
        self.screen = pygame.display.set_mode(SCREEN_RECT.size)
        pygame.display.set_caption("AIR FIGHT with two type of AI")

        # 2.创建游戏时钟
        # Create clock
        self.clock = pygame.time.Clock()

        # 3.创建游戏背景
        # Create background picture
        bg1 = BackGround()
        bg2 = BackGround(True)
        self.back_group = pygame.sprite.Group(bg1, bg2)

        # 4.创建游戏模式选择按钮
        # Create three buttons to choose game mode
        button_player = ButtonSprite("./image/button_player.png", 180, 100)
        button_flock = ButtonSprite("./image/button_flock.png", 180, 250)
        button_DRL = ButtonSprite("./image/button_DRL.png", 180, 400)
        self.button_group = pygame.sprite.Group(button_player, button_flock, button_DRL)
        self.button_list = [button_player, button_flock, button_DRL]

    # 选择游戏模式，监听鼠标点击事件
    # Choose game mode by listening to the mouse event
    def mouse_event_listener(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down_x, mouse_down_y = pygame.mouse.get_pos()

                if self.button_list[0].rect.collidepoint(mouse_down_x, mouse_down_y):
                    self.button_list[0].kill()
                    game = PlayerGame()
                    game.start_game()

                elif self.button_list[1].rect.collidepoint(mouse_down_x, mouse_down_y):
                    self.button_list[1].kill()
                    game = FlockGame()
                    game.start_game()

                elif self.button_list[2].rect.collidepoint(mouse_down_x, mouse_down_y):
                    self.button_list[2].kill()
                    game = DRLGame()
                    game.start_game()

    # 选择游戏模式
    # To choose game mode
    def choice_mode(self):
        while True:
            # 1.设置刷新频率
            # Set refresh frequency
            self.clock.tick(FRAME_PRE_SEC)

            # 2. 刷新屏幕
            # Refresh windows
            self.back_group.draw(self.screen)
            self.button_group.draw(self.screen)

            # 3. 监听鼠标事件
            # Listen to mouse
            self.mouse_event_listener()
            pygame.display.update()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    theGame = Game()
    theGame.choice_mode()