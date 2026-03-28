#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class SimpleExplorer:
    def __init__(self):
        rospy.init_node('simple_explorer', anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.twist = Twist()
        self.min_distance = 0.6  # 60cm de distance de sécurité

    def laser_callback(self, msg):
        # Le Lidar du Turtlebot scanne à 360°, où l'index 0 est exactement devant.
        # On va regarder un cône de +/- 30 degrés devant le robot.
        front_angles = list(msg.ranges[-30:]) + list(msg.ranges[:30])
        
        # On filtre les valeurs infinies (hors portée du lidar) ou parasites (0)
        clean_ranges = [r for r in front_angles if r > 0.1 and r < float('inf')]
        
        if clean_ranges and min(clean_ranges) < self.min_distance:
            # Obstacle détecté trop près : on s'arrête d'avancer et on pivote sur place
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5  # Tourner vers la gauche
        else:
            # Voie libre : on avance tout droit
            self.twist.linear.x = 0.15
            self.twist.angular.z = 0.0

        # On publie la commande
        self.pub.publish(self.twist)

if __name__ == '__main__':
    try:
        explorer = SimpleExplorer()
        rospy.loginfo("Mode exploration autonome démarré. (Ctrl+C pour arrêter)")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
