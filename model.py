import numpy as np

class Kmeans():
    def __init__(self, m=10):
        self.m = m
        self.n = None
        self.A = None # Número de hospitales (clusters)
        self.data = None # Número de casas
        self.hospitals = None # Almacenará los centroides (A, 2)
        self.clusters = None # Almacenará a qué hospital pertenece cada casa (n,)

    def set_m(self, m: int):
        """Define el tamaño de la cuadrícula y resetea el modelo."""
        self.m = m
        # Reseteamos todo si cambia el tamaño del grid
        self.n = None
        self.A = None
        self.data = None
        self.hospitals = None
        self.clusters = None
        print(f"Modelo reseteado con m={m}")
        return {"m": self.m, "message": "Grid m x m configurado."}

    def random_data_init(self, n: int):
        """Inicializa aleatoriamente la posición de los n datos (casas)"""
        if n is None:
            raise ValueError("Debe especificar el número de datos (n).")
        
        self.n = n
        
        if self.n > self.m * self.m:
            raise ValueError("El número de datos (n) no puede ser mayor que m*m.")

        # Lista con n datos aleatorios en un espacio 2D de tamaño m x m
        self.data = np.random.randint(0, self.m, (self.n, 2))
        return self.data
    
    def random_hospitals_init(self, A: int):
        """Inicializa aleatoriamente la posición de los A hospitales (centroides)"""
        if A is None:
            raise ValueError("Debe especificar el número de hospitales (A).")
        
        self.A = A
        
        if self.A > self.m * self.m:
            raise ValueError("El número de hospitales (A) no puede ser mayor que m*m.")
        
        # Lista con A hospitales aleatorios en un espacio 2D de tamaño m x m
        self.hospitals = np.random.randint(0, self.m, (self.A, 2))
        return self.hospitals
    
    def euclidian_distance(self, point1, point2):
        """Calcula la distancia euclidiana entre dos puntos en un espacio 2D"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def assign_clusters(self):
        """Asigna cada punto de datos al hospital más cercano"""
        if self.data is None or self.hospitals is None:
            raise ValueError("Debe inicializar los datos (casas) y los hospitales primero.")
        
        clusters = []
        for point in self.data:
            distances = [self.euclidian_distance(point, hospital) for hospital in self.hospitals]
            closest_hospital = np.argmin(distances)
            clusters.append(closest_hospital)
        
        self.clusters = np.array(clusters)
        return self.clusters
    
    def update_hospitals(self):
        """Actualiza la posición de los hospitales basándose en los puntos asignados"""
        if self.data is None or self.clusters is None:
            raise ValueError("Debe asignar clusters antes de actualizar los hospitales.")

        new_hospitals = []
        for i in range(self.A):
            # Obtenemos todas las 'casas' asignadas al hospital 'i'
            cluster_points = self.data[self.clusters == i]
            
            if len(cluster_points) > 0:
                # Calculamos el promedio (nuevo centroide)
                new_hospital = np.mean(cluster_points, axis=0)
            else:
                # Si un hospital no tiene casas, lo re-inicializamos aleatoriamente
                new_hospital = np.random.randint(0, self.m, 2)
            
            new_hospitals.append(new_hospital)
            
        self.hospitals = np.array(new_hospitals)
        return self.hospitals

    def calculate_metrics(self):
        """
        Calcula métricas de rendimiento del modelo actual.
        Devuelve:
            - average_distance: La distancia promedio de una casa a su hospital.
            - inertia: La suma de errores cuadráticos.
        """
        if self.data is None or self.hospitals is None or self.clusters is None:
            return {"average_distance": 0, "inertia": 0}
        
        total_distance = 0
        total_squared_error = 0 # Inertia
        
        # Recorremos cada casa para ver qué tan lejos está de su hospital
        for i, point in enumerate(self.data):
            cluster_idx = self.clusters[i] # El ID del hospital asignado
            centroid = self.hospitals[cluster_idx] # Las coordenadas de ese hospital
            
            # Usamos tu función existente para calcular la distancia
            dist = self.euclidian_distance(point, centroid)
            
            total_distance += dist
            total_squared_error += dist ** 2 # Inercia es distancia al cuadrado
            
        avg_distance = total_distance / self.n
        
        return {
            "average_distance": round(avg_distance, 2),
            "inertia": round(total_squared_error, 2)
        }