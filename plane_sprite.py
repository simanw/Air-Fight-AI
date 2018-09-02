import random
import pygame
import numpy
import math

# 屏幕大小常量
# Set screen size
SCREEN_RECT = pygame.Rect(0, 0, 480, 700)

# 每秒更新的帧数
# Set sections per second
FRAME_PRE_SEC = 100

# 创建定时器事件常量
CREATE_ENEMY_EVENT = pygame.USEREVENT

# 英雄发射子弹事件常量
HERO_FIRE_EVENT = pygame.USEREVENT + 1

# 敌机发射子弹事件常量
ENEMY_FIRE_EVENT = pygame.USEREVENT + 2


class GameSprite(pygame.sprite.Sprite):

    def __init__(self, image_name, speed=1):

        super().__init__()

        self.image = pygame.image.load(image_name)
        self.rect = self.image.get_rect()
        self.speed = speed

    def update(self):

        self.rect.y += self.speed


class ButtonSprite(GameSprite):
    def __init__(self, image_name, x, y):
        super().__init__(image_name)
        self.rect.x = x
        self.rect.y = y


class BackGround(GameSprite):
    """
    游戏背景精灵
    """
    def __init__(self, is_alt=False):

        # 调用父类方法实现父类的创建
        super().__init__("./image/background.png")

        # 判断是否为交替图像，如果是，需要设置初始位置
        if is_alt is True:
            self.rect.y = -self.rect.height

    def update(self):
        # 调用父类方法实现y轴滚动
        super().update()

        # 判断是否移出屏幕，如果移出屏幕，将图像设置到屏幕的上方
        if self.rect.y >= SCREEN_RECT.height:
            self.rect.y = -self.rect.height


class Enemy(GameSprite):

    def __init__(self):

        # 1.调用父类方法，创建敌机精灵，同时指定敌机图片
        # Create enemy sprite and load the picture
        super().__init__("./image/enemy0.png")

        # 2.指定敌机的初始随机速度
        # Set enemy's initial speed randomly
        self.speed = random.randint(2, 3)

        # 3.指定敌机的初始随机位置
        # Set enemy's horizontal position randomly
        self.rect.bottom = 0
        max_x = SCREEN_RECT.width - self.rect.width
        self.rect.x = random.randint(self.rect.width, max_x)

        # 4.创建敌机的子弹精灵组
        # Create this enemy's bullet group
        self.enemy_bullet_group = pygame.sprite.Group()

    def update(self):

        # 1.调用父类方法，保持垂直方向的飞行
        super().update()

        # 2.判断是否飞出屏幕，如果是，需要从精灵组删除敌机
        if self.rect.y >= SCREEN_RECT.height:
            # print("超出屏幕，从精灵组删除")
            self.kill()

    def fire(self):
        # 1.创建子弹精灵
        bullet = Bullet("./image/bullet1.png", self.speed+1)
        # 2.设置精灵的位置
        bullet.rect.bottom = self.rect.bottom + bullet.rect.height + 2
        bullet.rect.centerx = self.rect.centerx
        # 3.将精灵添加到精灵组
        self.enemy_bullet_group.add(bullet)

        # 敌机销毁前调用


class Bullet(GameSprite):
    def __init__(self, image_name, speed):

        # 调用父类方法，设置子弹图片，设置初始速度。
        super().__init__(image_name, speed)

    def update(self):

        # 调用父类方法，让子弹垂直方向飞行
        super().update()

        # 判断英雄子弹是否飞出屏幕上方 敌机子弹是否飞出屏幕下方
        if self.rect.bottom < 0 or self.rect.y > SCREEN_RECT.height:
            self.kill()


class Hero(GameSprite):
    """
    英雄精灵
    """
    def __init__(self):

        # 1.调用父类方法设置图片和速度
        super().__init__("./image/heroSmall.png", 0)
        self.xspeed = 0
        self.yspeed = 0
        # 2.设置英雄的初始位置
        self.rect.centerx = SCREEN_RECT.centerx
        self.rect.bottom = SCREEN_RECT.bottom - 50

        # 3.创建子弹的精灵组
        self.bullet_group = pygame.sprite.Group()

    def fire(self):
        # 同时发射两颗子弹
        # for i in range(2):

        # 1.创建子弹精灵
        bullet = Bullet("./image/bullet2.png", -2)
        # 2.设置精灵的位置
        bullet.rect.bottom = self.rect.y - 5
        bullet.rect.centerx = self.rect.centerx
        # 3.将精灵添加到精灵组
        self.bullet_group.add(bullet)


class PlayerHero(Hero):
    def __init__(self):
        super().__init__()

    # 英雄的移动
    # Move Hero
    def update(self):
        self.rect.x += self.xspeed
        self.rect.y += self.yspeed

        # 控制英雄不能离开屏幕
        # Hero cannot move outside screen
        if self.rect.left <= 0:
            self.rect.left = 0
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= SCREEN_RECT.bottom:
            self.rect.bottom = SCREEN_RECT.bottom
        if self.rect.right >= SCREEN_RECT.right:
            self.rect.right = SCREEN_RECT.right


class FlockHero(Hero):
    def __init__(self):
        super().__init__()

        # 最大速率
        # Set Hero's max speed
        self.maxspeed = 10

        # 对英雄产生威胁的危险距离
        # Set the dangerous distance from Hero's current position
        self.dangerous_dist = self.maxspeed * 10

        # 将英雄的位置和速度向量化
        # Transform the position and speed into vectors, for the sake of calculation convenience
        self.position = numpy.array(self.rect.center)
        self.velocity = numpy.array([self.xspeed, self.yspeed])

        # 英雄的加速度初始值为(0. 0.)
        # Set initial acceleration
        self.acceleration = numpy.zeros(2)

    def my_update(self, enemies, bullets):
        # rule-based algorithm
        self.flock(enemies, bullets)

        # Update Hero's velocity and position
        self.velocity = numpy.add(self.velocity, self.acceleration)
        self.position = numpy.add(self.position, self.velocity)
        self.rect.center = self.position

        # Reset acceleration to 0 each loop
        self.acceleration = numpy.zeros(2)
        self.velocity = numpy.zeros(2)

        # 控制英雄不能离开屏幕
        # Hero cannot move outside the screen
        if self.rect.x <= 0:
            self.rect.x = 0
        if self.rect.y <= 0:
            self.rect.y = 0
        if self.rect.bottom >= SCREEN_RECT.bottom:
            self.rect.bottom = SCREEN_RECT.bottom
        if self.rect.right >= SCREEN_RECT.right:
            self.rect.right = SCREEN_RECT.right
        self.position = numpy.array(self.rect.center)

    # We accumulate a new acceleration each time based on two rules
    def flock(self, enemies, bullets):
        # Enemies and enemies'bullets are seen as blocks to Hero
        blocks = pygame.sprite.Group(enemies, bullets)

        # Get avoidance preference
        avoid = self.avoidance(blocks)
        # Get attack preference
        attack = self.attack(enemies)
        # weight these two forces
        avoid = [1.5] * avoid
        attack = [1.0] * attack
        # add the force vectors to acceleration
        self.apply_acceleration(avoid)
        self.apply_acceleration(attack)

    def apply_acceleration(self, force):
        self.acceleration = numpy.add(self.acceleration, force)

    # 设置躲避规则
    # Set avoidance rule
    def avoidance(self, blocks):
        count = 0    # the number of blocks within dangerous scope
        steer = numpy.zeros(2)

        for block in blocks:
            block_position = [block.rect.x, block.rect.y]
            block_position = numpy.array(block_position)
            # 欧式距离
            # Get euclidean distance
            distance = numpy.linalg.norm(self.position - block_position)

            # This block is within the dangerous scope
            if distance > 0 and distance < self.dangerous_dist :
                difference = numpy.subtract(self.position, block_position)
                # weight by its distance from Hero 距离越远权重越小
                difference = self.normalize(difference) / [distance]
                steer = numpy.add(difference, steer)
                count += 1

        # 确定了加速度的方向后，下面确定加速度的大小，取平均值
        # Average -- divided by how many
        if count > 0:
            # 主要向水平方向躲避，因此延长水平移动距离
            x_direction_weight = self.maxspeed * steer[0]
            x_direction_weight = numpy.array([x_direction_weight, 0])
            steer = numpy.add(x_direction_weight, steer) / [count]

        # As long as magnitude of steer is greater than 0
        if self.magnitude(steer) > 0:
            # Implement Reynolds: Steering = Desired - Velocity
            steer = self.normalize(steer)
            temp = steer
            steer = steer * [self.maxspeed]
            if self.magnitude(numpy.add(steer, self.velocity)) > self.maxspeed:
                length = self.get_solution_of_equation(temp)
                steer = temp * [length]

        return steer


    # 攻击目标：水平距离在英雄攻击半径以内 and 水平距离最小
    # and 该敌机距离英雄最近的子弹的速率半径圆与英雄的速率半径圆相离
    # Set attack rule
    # Which enemy is the attack target?
    # Its horizontal distance from Hero is lower than Hero's attack radius and lowest among that of all enemies
    # and its bullet which is closest to Hero will not destroy Hero next update
    def attack(self, enemies):

        # 英雄的攻击半径为英雄的最大速度 乘以一个参数
        hero_attack_radius = self.maxspeed * 15

        # 处在英雄当前的攻击圈内的敌机个数
        # The number of enemies that locate within the attack circle
        count = 0

        # The minimum horizontal distance from Hero
        min_x_distance = SCREEN_RECT.width
        for enemy in enemies:
            enemy_position = numpy.array([enemy.rect.x, enemy.rect.y])
            # 敌机与英雄的水平距离 fabs()取绝对值
            x_distance = numpy.fabs(numpy.subtract(self.position, enemy_position)[0])
            # 为了减少计算，当敌机处在攻击圈内时，再看它的子弹距离
            if x_distance < hero_attack_radius:
                nearest_bullet_distance = SCREEN_RECT.width
                bullet_radius = 0
                for bullet in enemy.enemy_bullet_group:
                    bullet_position = numpy.array([bullet.rect.x, bullet.rect.y])
                    bullet_distance = numpy.linalg.norm(self.position - bullet_position)
                    # 由于子弹总是竖直向下运动的，可把子弹的速度当做标量
                    bullet_radius = bullet.speed
                    if bullet_distance < nearest_bullet_distance:
                        nearest_bullet_distance = bullet_distance

                if x_distance < min_x_distance and nearest_bullet_distance > self.maxspeed + bullet_radius:
                    min_x_distance = x_distance
                    count += 1
                    # 英雄水平移动
                    # To kill this block, Hero only need to move horizontally
                    desire = numpy.array([enemy.rect.x, self.position[1]])
                    steer = numpy.subtract(desire, self.position)
                    temp = self.normalize(steer)
                    if self.magnitude(numpy.add(steer, self.velocity)) > self.maxspeed:
                        length = self.get_solution_of_equation(temp)
                        steer = temp * [length]

        if count > 0:
            return steer
        # 没有符合条件的敌机，放弃攻击
        # Abandon attacking when there is no proper attack target
        elif count == 0:
            return numpy.zeros(2)

    # 将向量化为单位向量
    # transform input vector into unit vector
    def normalize(self, vector):
        magnitude = numpy.sqrt(numpy.vdot(vector, vector))
        if magnitude != 0:
            vector = vector / [magnitude]
            return vector
        else:
            return 0

    # 计算向量的模
    # Calculate the magnitude of a vector
    def magnitude(self, vector):
        return numpy.sqrt(numpy.vdot(vector, vector))

    def get_solution_of_equation(self, direction):
        x = direction[0]
        y = direction[1]

        a = (pow(x, 2) + pow(y, 2))
        b = 2 * (x * self.velocity[0] + y * self.velocity[1])
        c = pow(self.velocity[0], 2) + pow(self.velocity[1], 2) - pow(self.maxspeed, 2)
        if pow(b, 2) - 4 * a * c >= 0:
            z = math.sqrt(pow(b, 2) - 4 * a * c)
            return (-b+z)/2*a
        else:
            return 0


class DRLHero(Hero):
    def __init__(self):
        super().__init__()
        # 移动一次的速率
        # Set Hero's speed
        self.speed = 10
        self.dangerous_dist = self.rect.height * 5

    def my_update(self, action):
        if action == 0:
            self.rect.centery -= self.speed
        elif action == 1:
            self.rect.centery += self.speed
        elif action == 2:
            self.rect.centerx -= self.speed
        elif action == 3:
            self.rect.centerx += self.speed
        else:  # 不动 no move
            pass

        # 控制英雄不能离开屏幕
        # Hero cannot move outside the screen
        if self.rect.x <= 0:
            self.rect.x = 0
        if self.rect.y <= 0:
            self.rect.y = 0
        if self.rect.bottom >= SCREEN_RECT.bottom:
            self.rect.bottom = SCREEN_RECT.bottom
        if self.rect.right >= SCREEN_RECT.right:
            self.rect.right = SCREEN_RECT.right

    def bullet_above(self, enemies, bullets):

        # 子弹是否会打到英雄

        for bullet in bullets:
            if (self.rect.top - bullet.rect.bottom) < self.dangerous_dist and (bullet.rect.left <= self.rect.right or bullet.rect.right >= self.rect.left):
                return True

        # 敌机是否会打到英雄
        # Whether there exist a enemy above Hero going to collide with Hero,
        # in this case, the enemy can be seen as a bullet
        for enemy in enemies:
            if (self.rect.top - enemy.rect.bottom) < self.dangerous_dist and (enemy.rect.left <= self.rect.right or enemy.rect.right >= self.rect.left):
                return True

        return False

    # 敌机正上方有无敌机，因为子弹是向正前方射出
    def enemy_above(self, enemies):
        for enemy in enemies:
            if enemy.rect.left < self.rect.centerx < enemy.rect.right:
                return True

        return False



