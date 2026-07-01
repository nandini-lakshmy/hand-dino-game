import pygame
import cv2
import mediapipe as mp
import numpy as np
import random

# ================= HAND TRACKING =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_y = None
jump_signal = False

# --- FIX: jump cooldown ---
JUMP_COOLDOWN = 20
jump_cooldown_timer = 0

# ================= PYGAME SETUP =================
pygame.init()
WIDTH, HEIGHT = 1000, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Controlled Dino Game")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 36)
big_font = pygame.font.SysFont(None, 72)

# ================= DINO =================
dino_x = 80
dino_y = 280
dino_w, dino_h = 60, 65
velocity_y = 0
gravity = 1
jump_strength = -15
ground = 280

def draw_dino(x, y):
    pygame.draw.rect(screen, (60, 60, 60), (x, y + 20, 50, 30))
    pygame.draw.rect(screen, (60, 60, 60), (x + 35, y, 25, 25))
    pygame.draw.circle(screen, (255, 255, 255), (x + 52, y + 10), 3)
    pygame.draw.rect(screen, (60, 60, 60), (x + 5, y + 50, 10, 15))
    pygame.draw.rect(screen, (60, 60, 60), (x + 30, y + 50, 10, 15))
    pygame.draw.polygon(screen, (60, 60, 60),
                        [(x - 15, y + 35), (x, y + 30), (x, y + 45)])

# ================= OBSTACLES =================
obs_w, obs_h = 30, 40
speed = 8

obstacles = [
    {"x": WIDTH + 200},
    {"x": WIDTH + 500},
    {"x": WIDTH + 800},
]

# ================= GAME STATE =================
score = 0
game_over = False
collision_timer = 0
GAME_OVER_DELAY = 90

def reset_game():
    global dino_y, velocity_y, score, game_over, collision_timer, obstacles
    dino_y = ground
    velocity_y = 0
    score = 0
    game_over = False
    collision_timer = 0
    obstacles = [
        {"x": WIDTH + 200},
        {"x": WIDTH + 500},
        {"x": WIDTH + 800},
    ]

reset_game()
running = True

# ================= MAIN LOOP =================
while running:
    clock.tick(60)
    screen.fill((255, 255, 255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ---------- CAMERA ----------
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not game_over and result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            y = hand.landmark[8].y  # index finger

            if prev_y is not None and y < prev_y - 0.03 and jump_cooldown_timer == 0:
                jump_signal = True
                jump_cooldown_timer = JUMP_COOLDOWN

            prev_y = y

    # ---------- GAME LOGIC ----------
    if jump_cooldown_timer > 0:
        jump_cooldown_timer -= 1

    if not game_over:
        if jump_signal and dino_y == ground:
            velocity_y = jump_strength
            jump_signal = False

        velocity_y += gravity
        dino_y += velocity_y

        if dino_y >= ground:
            dino_y = ground
            velocity_y = 0

        for obs in obstacles:
            obs["x"] -= speed
            if obs["x"] < -obs_w:
                obs["x"] = WIDTH + random.randint(300, 500)
                score += 1

        dino_rect = pygame.Rect(dino_x, dino_y, dino_w, dino_h)

        for obs in obstacles:
            obs_rect = pygame.Rect(obs["x"], 300, obs_w, obs_h)
            if dino_rect.colliderect(obs_rect):
                game_over = True
                collision_timer = GAME_OVER_DELAY

    else:
        collision_timer -= 1
        if collision_timer <= 0:
            reset_game()

    # ---------- DRAW ----------
    draw_dino(dino_x, dino_y)

    for obs in obstacles:
        pygame.draw.rect(screen, (255, 0, 0),
                         (obs["x"], 300, obs_w, obs_h))

    pygame.draw.line(screen, (0, 0, 0), (0, 350), (500, 350), 2)

    screen.blit(font.render(f"Score: {score}", True, (0, 0, 0)), (20, 20))

    if game_over:
        screen.blit(big_font.render("GAME OVER", True, (200, 0, 0)),
                    (120, 150))

    # ---------- CAMERA INSIDE GAME ----------
    cam = cv2.resize(frame, (480, 360))
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    cam = np.transpose(cam, (1, 0, 2))
    screen.blit(pygame.surfarray.make_surface(cam), (520, 20))

    pygame.display.update()

# ================= CLEAN EXIT =================
cap.release()
pygame.quit()