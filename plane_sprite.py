import random
import pygame
import numpy
import math

# Set screen size
SCREEN_RECT = pygame.Rect(0, 0, 480, 700)

# Set sections per second
FRAME_PRE_SEC = 100

# 创建定时器事件常量
# Create event of enemies generation
CREATE_ENEMY_EVENT = pygame.USEREVENT

# 英雄发射子弹事件常量
# Create event of hero fires
HERO_FIRE_EVENT = pygame.USEREVENT + 1

# 敌机发射子弹事件常量
# Create event of enemies fires
ENEMY_FIRE_EVENT = pygame.USEREVENT + 2

# Original class
class GameSprite(pygame.sprite.Sprite):

    def __init__(self, image_name, speed=1):

        super().__init__()

        self.image = pygame.image.load(image_name)
        self.rect = self.image.get_rect()
        self.speed = speed

    def update(self):

        self.rect.y += self.speed


class ButtonSprite(GameSprite):
    """
    Buttons in the panel of game mode choice
    """
    def __init__(self, image_name, x, y):
        super().__init__(image_name)
        self.rect.x = x
        self.rect.y = y


class BackGround(GameSprite):
    """
    Game background
    """
    def __init__(self, is_alt=False):

        # Create background sprite
        super().__init__("./image/background.png")

        # 判断是否为交替图像，如果是，需要设置初始位置
        if is_alt is True:
            self.rect.y = -self.rect.height

    # Roll the background in Y axis
    def update(self):
        
        super().update()

        # 判断是否移出屏幕，如果移出屏幕，将图像设置到屏幕的上方
        if self.rect.y >= SCREEN_RECT.height:
            self.rect.y = -self.rect.height


class Enemy(GameSprite):

    def __init__(self):

        # Create enemy sprite and load the picture
        super().__init__("./image/enemy0.png")

        # Set enemy's initial speed randomly
        self.speed = random.randint(2, 3)

        # Set enemy's horizontal position randomly
        self.rect.bottom = 0
        max_x = SCREEN_RECT.width - self.rect.width
        self.rect.x = random.randint(self.rect.width, max_x)

        # Create this enemy's bullet group
        self.enemy_bullet_group = pygame.sprite.Group()

    def update(self):

        # Enemies fly in Y axis
        super().update()

        # Delete an enemy when it leaves the screen
        if self.rect.y >= SCREEN_RECT.height:
            self.kill()

    def fire(self):
        # Create a bullet of an enemy
        bullet = Bullet("./image/bullet1.png", self.speed+1)
        # Set position of the bullet
        bullet.rect.bottom = self.rect.bottom + bullet.rect.height + 2
        bullet.rect.centerx = self.rect.centerx
        # Add created bullted in this enemy's bullet group
        self.enemy_bullet_group.add(bullet)


class Bullet(GameSprite):
    def __init__(self, image_name, speed):

        super().__init__(image_name, speed)

    def update(self):

        # Bullets fly in Y axis
        super().update()

        # Delete an enemy's bullet or a hero's bullet
        if self.rect.bottom < 0 or self.rect.y > SCREEN_RECT.height:
            self.kill()


class Hero(GameSprite):
    """
    Hero
    """
    def __init__(self):

        super().__init__("./image/heroSmall.png", 0)
        self.xspeed = 0
        self.yspeed = 0
        # Set hero's initial position
        self.rect.centerx = SCREEN_RECT.centerx
        self.rect.bottom = SCREEN_RECT.bottom - 50

        # Create hero's bullet group
        self.bullet_group = pygame.sprite.Group()

    def fire(self):
        
        bullet = Bullet("./image/bullet2.png", -2)
        # Set a bullet's position
        bullet.rect.bottom = self.rect.y - 5
        bullet.rect.centerx = self.rect.centerx
        # Add created bullet in hero's bullet group
        self.bullet_group.add(bullet)


class PlayerHero(Hero):
    """
    Control hero by real player
    """
    def __init__(self):
        super().__init__()

    # Move Hero
    def update(self):
        self.rect.x += self.xspeed
        self.rect.y += self.yspeed

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
    """
    Adapt flock algorithm to control hero
    """
    def __init__(self):
        super().__init__()

        # Set Hero's max speed
        self.maxspeed = 10

        # Set the dangerous distance from Hero's current position
        self.dangerous_dist = self.maxspeed * 10

        # Transform the position and speed into vectors, for the sake of calculation convenience
        self.position = numpy.array(self.rect.center)
        self.velocity = numpy.array([self.xspeed, self.yspeed])

        # Set initial acceleration, (0, 0)
        self.acceleration = numpy.zeros(2)

    def my_update(self, enemies, bullets):

        # flock-based algorithm. 
        self.flock(enemies, bullets)

        # Update Hero's velocity and position
        self.velocity = numpy.add(self.velocity, self.acceleration)
        self.position = numpy.add(self.position, self.velocity)
        self.rect.center = self.position

        # Reset acceleration to 0 each loop
        self.acceleration = numpy.zeros(2)
        self.velocity = numpy.zeros(2)

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


    def flock(self, enemies, bullets):
        """
        We accumulate a new acceleration each time based on two rules
        """
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

    # Set avoidance rule
    def avoidance(self, blocks):
        count = 0    # the number of blocks within dangerous scope
        steer = numpy.zeros(2)

        # Calculate the direction of avoidance vector 
        for block in blocks:
            block_position = [block.rect.x, block.rect.y]
            block_position = numpy.array(block_position)
            
            # Get euclidean distance between hero and a block
            distance = numpy.linalg.norm(self.position - block_position)

            # This block is within the dangerous scope
            if distance > 0 and distance < self.dangerous_dist :
                difference = numpy.subtract(self.position, block_position)
                # weight by its distance from Hero; the farer a block is, the smaller its weight is.
                difference = self.normalize(difference) / [distance]
                steer = numpy.add(difference, steer)
                count += 1

        # Calculate the size of avoidance vector
        if count > 0:
            # 主要向水平方向躲避，因此延长水平移动距离
            # Prefer to avoid horizonally, so 
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


    # Set attack rule
    # Which enemy is the target to be attacked?
    # Its horizontal distance from Hero is lower than Hero's attack radius and lowest among that of all enemies
    # and its bullet which is closest to Hero will not destroy Hero next update
    def attack(self, enemies):

        # Set attack radius
        hero_attack_radius = self.maxspeed * 15

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
        # Abandon attacking when there is no proper target to be attacked
        elif count == 0:
            return numpy.zeros(2)

    # 将向量化为单位向量
    # Transform input vector into unit vector
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

    # Get the solution of a binary equation
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
        # Set Hero's speed and dangerous distance
        self.speed = 10
        self.dangerous_dist = self.rect.height * 5

    def my_update(self, action):
        if action == 1: # move down
            self.rect.centery -= self.speed
        elif action == 2:  # move up
            self.rect.centery += self.speed
        elif action == 3:  # move left
            self.rect.centerx -= self.speed
        elif action == 4:  # move right
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
        # Judje whether there exists a enemy above Hero going to collide with Hero,
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



