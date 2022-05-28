def ChooseArea(img = None):
    """
    This function display image for user to choose areas which they want to delete.
    Args:
        img: Input image. Defaults to None.

    Returns:
        mask: 2D Binary mask. 
    """
    import pygame
    import cv2
    import numpy as np

    pygame.init()

    # CHANGE INPUT IMAGE HERE IF YOU WANT
    # img = cv2.imread("./img.jpg", 1)

    # DO NOT TOUCH THIS!!!
    img_rot = np.rot90(img)
    SCREEN_WIDTH = img_rot.shape[0]
    SCREEN_HEIGHT = img_rot.shape[1]

    pygame.display.set_caption("Seam Carving")
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    background_surf = pygame.surfarray.make_surface(img_rot).convert()
    background_surf = pygame.transform.flip(background_surf, True, False)

    fps = 120   
    clock = pygame.time.Clock()

    start = True
    flag = False
    pts = []

    # OUR OUTPUT VARAIABLE
    mask = np.zeros((img.shape[0], img.shape[1]))

    while start:
        window.blit(background_surf, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                start = False
                pygame.quit()
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                flag = True
                print("Down")
                x, y = pygame.mouse.get_pos()
                pygame.draw.rect(background_surf, (255, 0, 0), pygame.Rect(x, y, 1, 1))
                pts.append(pygame.mouse.get_pos())
            elif flag and event.type == pygame.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                pygame.draw.rect(background_surf, (255, 0, 0), pygame.Rect(x, y, 1, 1))
                pts.append(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                print("Up")
                npar = np.array(pts)
                pygame.draw.polygon(background_surf, (255, 255, 255), pts)
                
                cv2.fillConvexPoly(mask, npar, 1)
                            
                # WE GOT THE MASK!!! DO ANYTHING WITH IT
                cv2.imshow('mask', mask)
                cv2.waitKey(0)
                #######################################
                pts = []
                flag = False

        pygame.display.update()
        clock.tick(fps)
    return mask
