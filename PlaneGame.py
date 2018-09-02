# -*- coding:utf-8 -*-
# __author__ = 'wsm'
from pygame.locals import *
from plane_sprite import *
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import pygame
import numpy as np

EPISODES = 2000


class PlaneGame(object):
    """
    飞机大战主类 main class
    """
    def __init__(self):
        # 创建游戏窗口
        # Create game windows
        self.screen = pygame.display.set_mode(SCREEN_RECT.size, HWSURFACE | DOUBLEBUF)
        pygame.display.set_caption("Air Fight with two type of AI")

        # 设置时钟及其刷新频率
        # Set clock and its refresh frequency
        self.clock = pygame.time.Clock()
        self.clock.tick(FRAME_PRE_SEC)

        # 实时显示得分
        # Show the score instantly
        pygame.font.init()
        self.score = 0
        self.score_font = pygame.font.SysFont('arial', 16)

        # 调用私有方法，创建精灵和精灵组
        # Create sprites and sprite groups by private method
        self.__create_sprites()

        # 设置定时器事件 - 创建敌机 2s，英雄发射子弹1.5ms, 敌机发射子弹3s
        # Set three events to appear on the event queue:
        # 1. create one enemy per 2s
        # 2. hero fires per 1.5s
        # 3. enemy fires per 3s
        pygame.time.set_timer(CREATE_ENEMY_EVENT, 2000)
        pygame.time.set_timer(HERO_FIRE_EVENT, 1500)
        pygame.time.set_timer(ENEMY_FIRE_EVENT, 3000)

    def __create_sprites(self):

        # 创建游戏背景
        # Create background sprite
        bg1 = BackGround()
        bg2 = BackGround(True)
        self.back_group = pygame.sprite.Group(bg1, bg2)

        # 创建敌机精灵组
        # Create enemy sprite group
        self.enemy_group = pygame.sprite.Group()

    def _check_collide(self):
        # 1.英雄子弹摧毁敌机
        # Hero's bullets destroy enemies
        enemies = pygame.sprite.groupcollide(self.hero.bullet_group, self.enemy_group, True, True)
        for enemy in enemies:
            self.score += 1

        # 2.敌机或者子弹撞毁英雄
        # Hero is killed by enemies or enemies'bullets
        enemies_killers = pygame.sprite.spritecollide(self.hero, self.enemy_group, True)
        bullets = 0
        for one_enemy in self.enemy_group:
            bullets += len(pygame.sprite.spritecollide(self.hero, one_enemy.enemy_bullet_group, False))

        # 判断列表有无内容
        # If there exist enemy or bullet which collide with Hero, the game is over
        if len(enemies_killers) > 0 or bullets > 0:
            # 销毁英雄
            self.hero.kill()
            # 结束游戏
            PlaneGame._game_over(self)

    def _update_sprite(self):
        self.back_group.update()
        self.back_group.draw(self.screen)

        self.enemy_group.update()
        self.enemy_group.draw(self.screen)

        # 所有敌机的子弹都要更新
        # All of bullets of each enemy need to be updated their location
        for one_enemy in self.enemy_group:
            one_enemy.enemy_bullet_group.update()
            one_enemy.enemy_bullet_group.draw(self.screen)

        self.hero_group.update()
        self.hero_group.draw(self.screen)

        self.hero.bullet_group.update()
        self.hero.bullet_group.draw(self.screen)

    @staticmethod
    def _game_over(self):
        pygame.quit()
        exit()


# Real player class
class PlayerGame(PlaneGame):

    def __init__(self):
        super().__init__()
        self.hero = PlayerHero()
        self.hero_group = pygame.sprite.Group(self.hero)

    def start_game(self):
        # 为了让窗口长时间显示，需要放到循环里
        # To display the windows constantly, we need to update the windows in a loop
        while True:
            # 检测真人操作
            # Detect the player's manipulation
            self._event_handler()

            # 碰撞检测
            # Detect collision
            super()._check_collide()

            # 更新/绘制精灵组
            # Update sprites on the display
            super()._update_sprite()

            # 实时显示得分
            # Show the score instantly
            self.score_surface = self.score_font.render(u'score = %d' % self.score, True, (0, 0, 0))
            self.screen.blit(self.score_surface, (5, 5))

            # 更新显示
            pygame.display.update()

    def _event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                super()._game_over(self)

            elif event.type == CREATE_ENEMY_EVENT:
                # 敌机出场
                # Create one enemy
                enemy = Enemy()
                # This enemy fires
                enemy.fire()
                # Add this enemy to the enemy sprite group
                self.enemy_group.add(enemy)

            elif event.type == HERO_FIRE_EVENT:
                self.hero.fire()

            elif event.type == ENEMY_FIRE_EVENT:
                for one_enemy in self.enemy_group:
                    one_enemy.fire()

        # 使用键盘提供的方法获取键盘按键 - 按键元组
        # Get the press tuple from the keyboard
        pressed_key = pygame.key.get_pressed()

        # 判断元组对应的按键索引值 按下为1，否则为0
        # If the  key is pressed, corresponding value in the press tuple is 1, otherwise is 0;
        # 速度=4个像素/每次刷新
        # speed = 4 px/update
        if pressed_key[pygame.K_RIGHT]:
            self.hero.xspeed = 4
        elif pressed_key[pygame.K_LEFT]:
            self.hero.xspeed = -4
        elif pressed_key[pygame.K_UP]:
            self.hero.yspeed = -4
        elif pressed_key[pygame.K_DOWN]:
            self.hero.yspeed = 4
        else:
            self.hero.xspeed = 0
            self.hero.yspeed = 0


# Rule-based AI
# Reference on flock algorithm
class FlockGame(PlaneGame):

    def __init__(self):
        super().__init__()

        # Create Rule-based AI Hero
        self.hero = FlockHero()
        self.hero_group = pygame.sprite.Group(self.hero)

    def start_game(self):
        while True:
            pygame.display.update()
            # AI control Hero
            self.control_flight()

            # 碰撞检测
            super()._check_collide()

            # 更新/绘制精灵组
            super()._update_sprite()

            # 显示得分
            self.score_surface = self.score_font.render(u'score = %d' % self.score, True, (0, 0, 0))
            self.screen.blit(self.score_surface, (5, 5))

            # 更新显示
            pygame.display.update()

    def control_flight(self):
        # 根据事件队列创建精灵
        # Create sprites according to events queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                super()._game_over(self)
            elif event.type == CREATE_ENEMY_EVENT:
                # 敌机出场
                # Create enemy
                enemy = Enemy()
                enemy.fire()
                self.enemy_group.add(enemy)

            elif event.type == HERO_FIRE_EVENT:
                self.hero.fire()

            elif event.type == ENEMY_FIRE_EVENT:
                for one_enemy in self.enemy_group:
                    one_enemy.fire()

        # 获得所有敌机产生的所有子弹
        # Get all the bullets produced by all the enemies
        all_bullets = pygame.sprite.Group()
        for enemy in self.enemy_group:
            all_bullets.add(enemy.enemy_bullet_group)

        # 英雄根据当前环境做出反应
        self.hero.my_update(self.enemy_group, all_bullets)


class DRLGame(PlaneGame):

    def __init__(self):
        super().__init__()
        # 计算量太大，将事件频率降低，画面会好一点
        # As model-training is time-consuming, it would better to decrease the event frequency
        pygame.time.set_timer(CREATE_ENEMY_EVENT, 5000)
        pygame.time.set_timer(HERO_FIRE_EVENT, 2000)
        pygame.time.set_timer(ENEMY_FIRE_EVENT, 5000)

        # AI agent has 4 states and 5 corresponding actions
        self.state_size = 4
        self.action_size = 5
        self.hero = DRLHero()
        self.hero_group = pygame.sprite.Group(self.hero)
        self.STATE_1 = numpy.array([1, 0, 0, 0]) # 上方有敌机 没有在危险区内的子弹 There exist enemies above Hero, but no dangerous bullets
        self.STATE_2 = numpy.array([0, 1, 0, 0]) # 上方有敌机，有在危险区内的子弹  There exist enemies above Hero, also dangerous bullets
        self.STATE_3 = numpy.array([0, 0, 1, 0]) # 上方无敌机，没有子弹 There exist no enemies above Hero, as well as no bullets
        self.STATE_4 = numpy.array([0, 0, 0, 1]) # 上方无子弹，但是敌机会击中英雄 There exist enemies that can collide with Hero

        # Training parameters
        self.done = False
        self.batch_size = 32
        self.memory = deque(maxlen=500)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.current_epsilon = open('./epsilon.txt', 'w+')

    def start_game(self):
        self.event_handler()
        all_bullets = pygame.sprite.Group()
        for enemy in self.enemy_group:
            all_bullets.add(enemy.enemy_bullet_group)

        # 下载训练好的模型的参数
        # Load the trained parameters
        # self.load("./save/flight-war-dqn.h5")

        # Get initial game state
        game_state = self.get_game_state(self.enemy_group, all_bullets)
        game_state = np.reshape(game_state, [1, self.state_size])
        # 训练模型
        # Training the model by Q learning
        self.q_learning(game_state)

        # 模型训练好，用下面的代码
        # After getting the trained parameters, remove the comments
        # while True:
        #     # 产生敌机、发射子弹
        #     self.event_handler()
        #     # 使用模型
        #     game_state = self.get_game_state(self.enemy_group, all_bullets)
        #     print(game_state)
        #     if game_state.shape != (1,self.state_size):
        #         game_state = numpy.reshape(game_state,[1,self.state_size])
        #     act_values = self.model.predict(game_state)
        #     print(numpy.argmax(act_values[0]))
        #     self.hero.my_update(numpy.argmax(act_values[0]))
        #     # 被击中的敌机从精灵组中删去
        #     enemies = pygame.sprite.groupcollide(self.hero.bullet_group, self.enemy_group, True, True)
        #     for enemy in enemies:
        #         self.score += 1
        #     # 碰撞检测
        #     #super()._check_collide()
        #     # 更新/绘制精灵组
        #     super()._update_sprite()
        #     # 显示得分
        #     self.score_surface = self.score_font.render(u'score = %d' % self.score, True, (0, 0, 0))
        #     self.screen.blit(self.score_surface, (5, 5))
        #     # 更新显示
        #     pygame.display.update()

    def event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                super()._game_over(self)
            elif event.type == CREATE_ENEMY_EVENT:
                # 敌机出场
                enemy = Enemy()
                enemy.fire()
                self.enemy_group.add(enemy)

            elif event.type == HERO_FIRE_EVENT:
                self.hero.fire()

            elif event.type == ENEMY_FIRE_EVENT:
                for one_enemy in self.enemy_group:
                    one_enemy.fire()

    def get_game_state(self, enemies, bullets):
        enemy_above = self.hero.enemy_above(enemies)
        bullet_above = self.hero.bullet_above(enemies, bullets)
        # 状态1：英雄上方有敌机无子弹
        # STATE_1
        if enemy_above == True and bullet_above == False:
            return numpy.array([1, 0, 0, 0])
        # 状态2：有敌机有子弹
        # STATE_2
        elif enemy_above == True and bullet_above == True:
            return numpy.array([0, 1, 0, 0])
        # 状态3：无敌机无子弹
        # STATE_3
        elif enemy_above == False and bullet_above == False:
            return numpy.array([0, 0, 1, 0])
        # 状态4： 没有英雄正对的敌机，但是该敌机会击毁英雄，因此可以看做是子弹
        # STATE_4
        elif enemy_above == False and bullet_above == True:
            return numpy.array([0, 0, 0, 1])

    def q_learning(self, state):
        for e in range(EPISODES):
            for time in range(200):
                action = self.act(state)
                next_state, reward, done, _ = self.step(action)
                # reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    self.update_target_model()
                    # print("episode: {}/{}, timescore: {}, e: {:.2}"
                    #      .format(e, EPISODES, time, self.epsilon))
                    break
                if len(self.memory) > self.batch_size:
                    self.replay(self.batch_size)
            # if e % 10 == 0:
            # self.save("./save/flight-war-dqn.h5")

    def _check_collide(self):
        # 即将与英雄碰撞的敌机
        # Get enemies which will collide with Hero
        enemies_killers = pygame.sprite.spritecollide(self.hero, self.enemy_group, True)

        bullets = 0
        for one_enemy in self.enemy_group:
            bullets += len(pygame.sprite.spritecollide(self.hero, one_enemy.enemy_bullet_group, False))

        if len(enemies_killers) > 0 or bullets > 0:
            # 销毁英雄
            # Hero will be killed
            return True
        else:
            # Hero is still alive
            return False

    def step(self, action):
        pygame.display.update()
        self.hero.my_update(action)

        # ------------------实时演示学习过程----------------------------------------
        # ------------------Show the training process instantly--------------------
        self.event_handler()
        # 被击中的敌机从精灵组中删去
        # Delete the destroyed enemies from enemy sprite group
        enemies = pygame.sprite.groupcollide(self.hero.bullet_group, self.enemy_group, True, True)

        for enemy in enemies:
            self.score += 1

        # 碰撞检测
        super()._check_collide()
        # 更新精灵
        super()._update_sprite()

        # 显示得分
        # Show the score instantly
        self.score_surface = self.score_font.render(u'score = %d' % self.score, True, (0, 0, 0))
        self.screen.blit(self.score_surface, (5, 5))

        # 更新
        pygame.display.update()
        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------

        all_bullets = pygame.sprite.Group()
        for enemy in self.enemy_group:
            all_bullets.add(enemy.enemy_bullet_group)

        # 获得下一个状态
        # Get next state
        next_state = self.get_game_state(self.enemy_group, all_bullets)

        # 英雄是否被摧毁
        # If Hero is killed, done is TRUE; otherwise done is FALSE
        done = self._check_collide()
        reward = 0.

        # 英雄没有阵亡
        # Hero is still alive
        if not done:
            reward += 1.0

        # 英雄上方有敌机，并且没有在危险区域内的子弹
        if (next_state==self.STATE_1).all():
            reward += 30.0

        # 英雄上方有敌机，并且有在危险区域内的子弹
        elif (next_state==self.STATE_2).all():
            reward += -10.0

        # 英雄上方无敌机，并且没有在危险区域内的子弹
        elif (next_state==self.STATE_3).all():
            reward += 5.0

        # 英雄上方没有子弹，但是因为敌机的size比子弹小，存在敌机会击毁英雄
        # Though there is no bullet above Hero, the enemy probably kills Hero because its smaller size than bullets
        # This is because the picture size of enemy is smaller than that of bullet
        elif (next_state==self.STATE_4).all():
            reward += -10.0

        # 英雄阵亡
        # Hero is dead
        elif done:
            reward += -30.0

        return next_state, reward, done, {}

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 根据当前状态做出动作
    # Act according to current state
    def act(self, state):
        # 以epsilon的概率探索
        # Act randomly with the possibility of epsilon
        if numpy.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # 采用模型预测的结果
        # Act according to the prediction with the possibility of (1-epsilon)
        act_values = self.model.predict(state)
        return numpy.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if state.shape != (1, self.state_size):
                state = numpy.reshape(state, [1, self.state_size])
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * numpy.amax(t)
                # target[0][action] = reward + self.gamma * t[numpy.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.current_epsilon.write(str(self.epsilon))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)