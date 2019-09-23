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
FRAME_PER_ACTION = 1

GAME = 'flight'


class PlaneGame(object):
    """
    Game main class
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
        self._create_sprites()

        # 设置定时器事件 - 创建敌机 2s，英雄发射子弹1.5ms, 敌机发射子弹3s
        # Set three events to appear on the event queue:
        # 1. create one enemy per 2s
        # 2. hero fires per 1.5s
        # 3. enemy fires per 3s
        pygame.time.set_timer(CREATE_ENEMY_EVENT, 2000)
        pygame.time.set_timer(HERO_FIRE_EVENT, 1500)
        pygame.time.set_timer(ENEMY_FIRE_EVENT, 3000)

    def _start_game(self):
        # 为了让窗口长时间显示，需要放到循环里
        # To display the windows constantly, we need to update the windows in a loop
        while True:
            
            self._event_handler()

            self._control()

            # 碰撞检测
            # Detect collision
            self._check_collide()

            # 更新/绘制精灵组
            # Update sprites on the display
            self._update_sprite()

            # 实时显示得分
            # Show the score instantly
            self.score_surface = self.score_font.render(u'score = %d' % self.score, True, (0, 0, 0))
            self.screen.blit(self.score_surface, (5, 5))

            # 更新显示
            pygame.display.update()

    def _create_sprites(self):

        # 创建游戏背景
        # Create background sprite
        bg1 = BackGround()
        bg2 = BackGround(True)
        self.back_group = pygame.sprite.Group(bg1, bg2)

        # 创建敌机精灵组
        # Create enemy sprite group
        self.enemy_group = pygame.sprite.Group()

    def _event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._game_over(self)

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

    def _control(self):
        pass


    def _check_collide(self):
        # 1.英雄子弹摧毁敌机
        # 1. Hero's bullets destroy enemies
        enemies = pygame.sprite.groupcollide(self.hero.bullet_group, self.enemy_group, True, True)
        for enemy in enemies:
            self.score += 1

        # 2.敌机或者子弹撞毁英雄
        # 2. Hero is killed by enemies or enemies'bullets
        enemies_killers = pygame.sprite.spritecollide(self.hero, self.enemy_group, True)
        bullets = 0
        for one_enemy in self.enemy_group:
            bullets += len(pygame.sprite.spritecollide(self.hero, one_enemy.enemy_bullet_group, False))

        # 判断列表有无内容
        # If there exist enemy or bullet which collide with Hero, the game is over
        if len(enemies_killers) > 0 or bullets > 0:
            # Destroy hero
            self.hero.kill()
            # Game over
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
        super()._start_game()
        
    def _control(self):
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
# Adapted from flock algorithm
class FlockGame(PlaneGame):

    def __init__(self):
        super().__init__()

        # Create Rule-based AI Hero
        self.hero = FlockHero()
        self.hero_group = pygame.sprite.Group(self.hero)

    def start_game(self):
        super()._start_game()

    def _control(self):
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

        self.OBSERVE = 3200 # timesteps to observe before training
        self.EXPLORE = 2000000 # frames over which to anneal epsilon
        self.FINAL_EPSILON = 0.0001 # final value of epsilon
        self.INITIAL_EPSILON = 0.1 # starting value of epsilon
        self.FRAME_PER_ACTION = 1
        self.REPLAY_MEMORY = 500

        self.batch_size = 32
        self.gamma = 0.99  # decay rate of past observations
        self.learning_rate = 0.0001

        
    def start_game(self):

        super()._event_handler()
        self.model = self.build_model()
        # 训练模型
        # Training the model by Q learning
        self.train_network()

        # 模型训练好，用下面的代码
        # After getting the trained parameters, remove the comments
        # 下载训练好的模型的参数
        # Load the trained parameters
        # self.load("./save/model.h5")


    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        print("We finish building the model")
        return model

    def train_network(self):
        # store the previous observations in replay memory
        memory = deque()
        # get the first state by doing nothing
        do_nothing = np.zeros([self.action_size])
        s_t, r_0, terminal = self.step(do_nothing)
        # In keras, need to reshape
        s_t = np.reshape(s_t, [1, self.state_size])

        # start training
        epsilon = self.INITIAL_EPSILON
        t = 0

        while True:
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            a_t = np.zeros([self.action_size])
            #choose an action epsilon greedy
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(self.action_size)
                    a_t[action_index] = 1
                else:
                    q = self.model.predict(s_t)
                    max_Q = np.argmax(q)
                    action_index = max_Q
                    a_t[max_Q] = 1

            # We reduced the epsilon gradually
            if epsilon > self.FINAL_EPSILON and t > self.OBSERVE:
                epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE
            #run the selected action and observed next state and reward
            s_t1, r_t, terminal = self.step(a_t)
            s_t1 = np.reshape(s_t, [1, self.state_size])

            memory.append((s_t, action_index, r_t, s_t1, terminal))
            if len(memory) > self.REPLAY_MEMORY:
                memory.popleft()

            # only train if down observing
            if t > self.OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(memory, self.batch_size)

                # Now we do the experience replay
                state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                targets = self.model.predict(state_t)
                Q_sa = self.model.predict(state_t1)
                targets[range(self.batch_size), action_t] = reward_t + self.gamma*np.max(Q_sa, axis=1)*np.invert(terminal)

                loss += self.model.train_on_batch(state_t, targets)

            s_t = s_t1
            t += 1

            # save progress every 1000 iterations
            if t % 1000 == 0:
                print("Save model")
                self.model.save_weights("model.h5", overwrite=True)
            # print info
            state = ""
            if t <= self.OBSERVE:
                state = "observing"
            elif t > self.OBSERVE and t <= self.OBSERVE + self.EXPLORE:
                state = "exploring"
            else:
                state = "training"

            print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

        print("episode finished")
        print("***********************************")

    def step(self, action):
        super()._event_handler()
        pygame.display.update()

        self.hero.my_update(action)
        # 被击中的敌机从精灵组中删去
        # Delete the destroyed enemies from enemy sprite group
        enemies = pygame.sprite.groupcollide(self.hero.bullet_group, self.enemy_group, True, True)

        for enemy in enemies:
            self.score += 1

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
        # If Hero is killed, terminal is TRUE; otherwise terminal is FALSE
        terminal = self.is_terminal()
        reward = 0

        # 英雄没有阵亡
        # Hero is still alive
        if not terminal:
            reward += 1

        # 英雄上方有敌机，并且没有在危险区域内的子弹
        if (next_state==self.STATE_1).all():
            reward += 5

        # 英雄上方有敌机，并且有在危险区域内的子弹
        elif (next_state==self.STATE_2).all():
            reward += -5

        # 英雄上方无敌机，并且没有在危险区域内的子弹
        elif (next_state==self.STATE_3).all():
            reward += 1

        # 英雄上方没有子弹，但是因为敌机的size比子弹小，存在敌机会击毁英雄的可能
        # Though there is no bullet above Hero, the enemy probably kills Hero because its smaller size than bullets
        # This is because the picture size of enemy is smaller than that of bullet
        elif (next_state==self.STATE_4).all():
            reward += -5

        # 英雄阵亡
        # Hero is dead
        elif terminal:
            reward += -10

        return next_state, reward, terminal

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

    def is_terminal(self):
        # 敌机或者子弹撞毁英雄
        # Hero is killed by enemies or enemies'bullets
        enemies_killers = pygame.sprite.spritecollide(self.hero, self.enemy_group, True)
        bullets = 0
        for one_enemy in self.enemy_group:
            bullets += len(pygame.sprite.spritecollide(self.hero, one_enemy.enemy_bullet_group, False))

        # 判断列表有无内容
        # If there exist enemy or bullet which collide with Hero, the game is over
        return True if len(enemies_killers) > 0 or bullets > 0 else False



