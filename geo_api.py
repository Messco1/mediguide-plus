import requests

# ============================================================
# 1. Convertir un code postal en coordonnées GPS
# ============================================================
def get_coordinates(code_postal):
    url = f"https://api-adresse.data.gouv.fr/search/?q={code_postal}&type=municipality&limit=1"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        if data['features']:
            coords = data['features'][0]['geometry']['coordinates']
            city = data['features'][0]['properties']['label']
            return coords[1], coords[0], city  # lat, lon, ville
    except Exception as e:
        print(f"Erreur coordonnées : {e}")
    return None, None, None

# ============================================================
# 2. Chercher des médecins par spécialité près d'une position
# ============================================================
SPECIALTY_MAP = {
    'Surgery': 'chirurgien',
    'Cardiovascular / Pulmonary': 'cardiologue',
    'Neurology': 'neurologue',
    'Neurosurgery': 'neurochirurgien',
    'Orthopedic': 'orthopediste',
    'Psychiatry / Psychology': 'psychiatre',
    'Radiology': 'radiologue',
    'Urology': 'urologue',
    'Ophthalmology': 'ophtalmologue',
    'Dermatology': 'dermatologue',
    'Gastroenterology': 'gastro-enterologue',
    'Nephrology': 'nephrologue',
    'Hematology - Oncology': 'oncologue',
    'Endocrinology': 'endocrinologue',
    'General Medicine': 'medecin generaliste',
    'Emergency Room Reports': 'urgentiste',
    'Pediatrics - Neonatal': 'pediatre',
    'Obstetrics / Gynecology': 'gynécologue',
    'Dentistry': 'dentiste',
    'Podiatry': 'podologue',
    'Rheumatology': 'rhumatologue',
}

def find_doctors(specialty, lat, lon, rayon_km=10):
    spec_fr = SPECIALTY_MAP.get(specialty, 'medecin generaliste')
    
    url = "https://api.data.gouv.fr/api/1/datasets/annuaire-sante-medecins-generalistes/"
    
    # Utilise l'API Nominatim pour chercher des médecins autour
    search_url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={spec_fr}&format=json&limit=5"
        f"&viewbox={lon-0.1},{lat+0.1},{lon+0.1},{lat-0.1}"
        f"&bounded=1"
    )
    
    headers = {'User-Agent': 'MediGuide+ Student Project'}
    
    try:
        r = requests.get(search_url, headers=headers, timeout=5)
        results = r.json()
        
        doctors = []
        for res in results[:3]:
            doctors.append({
                'nom': res.get('display_name', 'Médecin').split(',')[0],
                'adresse': ', '.join(res.get('display_name', '').split(',')[:3]),
                'lat': float(res['lat']),
                'lon': float(res['lon']),
                'distance_km': round(
                    ((float(res['lat']) - lat)**2 + (float(res['lon']) - lon)**2)**0.5 * 111, 1
                )
            })
        
        doctors.sort(key=lambda x: x['distance_km'])
        return doctors
    
    except Exception as e:
        print(f"Erreur recherche médecins : {e}")
        return []

# ============================================================
# 3. Test complet
# ============================================================
if __name__ == '__main__':
    print("=== TEST COORDONNÉES ===")
    lat, lon, ville = get_coordinates("75001")
    print(f"Code postal 75001 → {ville} ({lat}, {lon})")

    print("\n=== TEST RECHERCHE MÉDECINS ===")
    doctors = find_doctors("Cardiovascular / Pulmonary", lat, lon)
    
    if doctors:
        for i, d in enumerate(doctors, 1):
            print(f"{i}. {d['nom']}")
            print(f"   Adresse : {d['adresse']}")
            print(f"   Distance : {d['distance_km']} km\n")
    else:
        print("Aucun médecin trouvé (normal selon la zone)")