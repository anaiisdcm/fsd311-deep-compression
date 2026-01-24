import subprocess

commandes = [
    "python pruning.py",
    "python weight_share.py saves/model_after_retraining.ptmodel",
    "python huffman_encode.py saves/model_after_weight_sharing.ptmodel"
]

for i, cmd in enumerate(commandes, start=1):
    # Vérifier que l'utilisateur veut bien lancer la commande
    reponse = input(f"Lancer la commande suivante ({i}/{len(commandes)}) : {cmd}\n(Y/n) : ").lower()
    if reponse != "y":
        print("Arrêt.")
        break

    print(f"\nExécution de l'étape {i} : {cmd}")

    # Exécution de la commande
    process = subprocess.Popen(
        cmd,
        shell=True
    )

    process.wait()

    # Vérifie si la commande a échoué
    if process.returncode != 0:
        print("Erreur lors de l'exécution")
        break