##############################################################################################################
################################################################################ IMPORTATION DES BIBLIOTHEQUES
##############################################################################################################
import streamlit as st 
import pandas as pd                                                      
import numpy as np
from sklearn.neighbors import NearestNeighbors
from streamlit_elements import elements, mui, html, sync
import math

##############################################################################################################
##################################################################################### CONFIGURATION DE LA PAGE
##############################################################################################################

st.set_page_config(
    page_title="Movie Recommendation Algorithm in Python",
    layout="wide"
)
st.sidebar.image('img/logo.png')

##############################################################################################################
################################################################################### DECLARATION DES CONSTANTES
##############################################################################################################
session = st.session_state
const_voisin = 100
url = 'https://image.tmdb.org/t/p/original'
url_get_imdb = 'https://www.imdb.com/title/'
btn = 0

##############################################################################################################
############################################################################# RECUPERATION DE LA PAGE COURANTE
##############################################################################################################
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
    if 'recherche' not in session:
        session['recherche'] = pd.DataFrame()
        session['recherche_f'] = pd.DataFrame()

##############################################################################################################
###################################################################################### IMPORTATION DES DONNEES
##############################################################################################################
films_db = pd.read_csv('data/films_db.csv')
films_db['#'] = films_db.index # créer une colonne qui duplique les index pour la recommandation
films_db['Genres'] = films_db['Genres'].str.split(',')
films_db['Annee'] = films_db["Annee"].str.slice(stop=4)
films_db['Annee'] = films_db['Annee'].astype(int)

##############################################################################################################
############################################################################ IMPORTATION DES DONNEES VECTORISE
##############################################################################################################
films_db_PCA = pd.read_csv('data/from_predict_PCA.csv')

##############################################################################################################
#################################################################################### IMPORTATION DES FONCTIONS
##############################################################################################################

def raccourcir_chaine(chaine):
    mots = chaine.split()
    if len(mots) <= 37:
        return chaine
    else:
        raccourci = ' '.join(mots[:37]) + '...'
        return raccourci
    
def verifGenres(dd):
    return any(genre in dd['Genres'] for genre in liked_genre)

def jaquette(i, url):
    if pd.notnull(i) and (i != 'False'):
        return url+i
    else:
        return 'img/not_found.png'
    
def overview(i):
    max_char = 50
    if pd.notnull(i) and (i != 'False'):
        return i
    else:
        return "There is no summary available for this movie"

def slideshow_swipeable(images, df_c):
        # Generate a session state key based on images.
        key = f"slideshow_swipeable_{str(images).encode().hex()}"

        # Initialize the default slideshow index.
        if key not in st.session_state:
            st.session_state[key] = 0

        # Get the current slideshow index.
        index = st.session_state[key]

        # Create a new elements frame.
        with elements(f"frame_{key}"):

            # Use mui.Stack to vertically display the slideshow and the pagination centered.

            with mui.Stack(spacing=1, alignItems="center"):

                with mui.SwipeableViews(index=index, resistance=True, onChangeIndex=sync(key)):

                    # Divide the images into chunks of 5.
                    image_chunks = [images[i:i+4] for i in range(0, len(images), 4)]

                    # Iterate over each chunk and display it in a separate slide.
                    for chunk in image_chunks:
  
                        with mui.Grid(container=True, spacing=1, justify_content="center"):
                            cp = 0
                            for image in chunk:
                                with mui.Grid(item=True, xs=3, sm=3, md=3, lg=3):
                                    lien = url_get_imdb+df_c.iloc[cp]['tconst']
                                    html.a(href=lien, target="_blank")(
                                        html.img(src=image, css={"width": "99%"})
                                    )
                                cp+=1


                def handle_change(event, value):
                    # Pagination starts at 1, but our index starts at 0, explaining the '-1'.
                    st.session_state[key] = value-1

                mui.Pagination(page=index+1, count=len(image_chunks), color="primary", onChange=handle_change)

def filtrer_films(df, ss_genres, ss_acteurs, ss_date, ss_note):
    # On transforme la colonne "Note_moyenne" en float
    df['Note_moyenne'] = df['Note_moyenne'].astype(float)

    # On filtre les films dont la note est supérieure ou égale à la note choisie
    df = df[df['Note_moyenne'] >= ss_note]

    # On filtre les films compris entre les deux dates choisies
    mask = (df['Annee'] >= ss_date[0]) & (df['Annee'] <= ss_date[1])
    df = df[mask]
    if ss_genres:
        df= df[df['Genres'].apply(lambda x: any(genres in ss_genres for genres in x))]
    
    if len(ss_acteurs) == 0:
        df['liste_acteurs'] = df['liste_acteurs'].str.split(',')
    else:
        # On filtre les films en gardant uniquement les acteurs choisis
        df= df[df['liste_acteurs'].apply(lambda x: any(actor in ss_acteurs for actor in x.split(',')))]
    
    return df

##############################################################################################################
############################################################################################### FILTRE SIDEBAR
##############################################################################################################
reco_by = st.sidebar.radio("", ['Acceuil', 'Par Genres', 'Par Films'], index=0)

if reco_by == 'Par Genres':
    genres_uniques = films_db["Genres"].explode().unique() # extraction des genres
    liked_genre = st.sidebar.selectbox('Select Genre', genres_uniques)
    btn = st.sidebar.button('Recommend') # Bouton pour lancer la recommandation ou le filtre
if reco_by == 'Par Films':
    selected_movie = st.sidebar.selectbox("Select a movie you like:", films_db['Titre'])
    btn = st.sidebar.button('Recommend') # Bouton pour lancer la recommandation ou le filtre
    
##################################################################################### RESET DES FILTRES ACTIFS
if reco_by == 'Acceuil':
    st.session_state['recherche'] = pd.DataFrame()
    st.session_state['recherche_f'] = pd.DataFrame()

##############################################################################################################
####################################################################################### SOUMISSION DES FILTRES
##############################################################################################################
if (btn) or (not st.session_state['recherche'].empty):
    st.subheader('We recommand:')
    ########################################################################################## SOUMISSION ALGO 
    if reco_by == 'Par Films':
        # DEBUT DE KNN
        X = films_db_PCA
        model = NearestNeighbors(n_neighbors=const_voisin, algorithm='auto')
        model.fit(X)
        idx = np.where(films_db['Titre'] == selected_movie)[0][0]
        distances, indices = model.kneighbors([X.iloc[idx]])
        # FIN DE KNN

        # DEBUT CREATION DES SESSIONS ET LISTE MOVIE RECO
        titres_films_proches = films_db.loc[indices[0]]
        first_row_index = titres_films_proches.index[0]
        session['recherche'] = titres_films_proches.drop(labels=first_row_index).reset_index(drop=True)
        session['recherche_f'] = titres_films_proches.drop(labels=first_row_index).reset_index(drop=True)
        # FIN DE CREATION DES SESSIONS ET LISTE MOVIE RECO

        # DEBUT AJOUTE LE FILMS CHOISI A LA SIDEBAR
        container = st.sidebar.container()
        container.image(jaquette(titres_films_proches.iloc[0]['poster_path'], url), width=250)
        container.title(titres_films_proches.iloc[0]['Titre'])
        container.write(overview(titres_films_proches.iloc[0]['overview']))
        # FIN DE AJOUT A LA SIDE BAR DU FILMS

        # DEBUT DE LA CREATION DES VARIABLES FILTRES SECONDAIRES
        ss_unique_genres = st.session_state.recherche_f["Genres"].explode().unique()
        ss_unique_acteurs = st.session_state.recherche_f["liste_acteurs"].str.split(',').explode().unique()
        ss_unique_date = st.session_state.recherche_f["Annee"].unique()
        max_note = int(session.recherche['Note_moyenne'].max())
        ss_unique_note = [math.floor(i) for i in range(max_note + 1)]
        # FIN DES CREATION DE VARIABLE FILTRES SECONDAIRES
        
        # DEBUT CREATION DES ELEMENTS FILTRE SECONDAIRE
        colx, colxx, colxxx, colxxxx = st.columns(4)
        with colx:
            ss_date = st.slider('Date',int(ss_unique_date.min()), int(ss_unique_date.max()), (int(ss_unique_date.min()), int(ss_unique_date.max())))
        with colxx:
            ss_genres = st.multiselect('Genres', ss_unique_genres, max_selections=3)
        with colxxx:
            ss_acteurs = st.multiselect('Acteurs', ss_unique_acteurs, max_selections=1 )
        with colxxxx:
            ss_note = st.selectbox('Note', ss_unique_note)
        # FIN DE CREATION DES ELEMENTS FILTRE SECONDAIRE

        ################################################################################# RECUPERATION DES FILTRES SECONDAIRES
        if ss_date or ss_genres or ss_acteurs or ss_note:
            ########################### VARIABLE DEBUG
            #st.write('-',ss_date)
            #st.write('-',ss_genres)
            #st.write('-',ss_acteurs)
            #st.write('-',ss_note)
            ####################### FIN VARIABLE DEBUG

            # MODIFICATION DE LA SESSION RECHERCHE_F
            st.session_state['recherche_f'] = filtrer_films(st.session_state['recherche'], ss_genres, ss_acteurs, ss_date, ss_note)

            ########################### VARIABLE DEBUG
            #st.write(st.session_state['recherche_f'].shape)
            #st.write(st.session_state['recherche_f'])
            #st.write(st.session_state['recherche'])
            ####################### FIN VARIABLE DEBUG

            ########################################################################## ON RECUPERE LA SESSION RECHERCHE ACTIVE
            if len(st.session_state['recherche_f']) == 100:
                len_df_f = len(st.session_state['recherche_f'])-1
            elif len(st.session_state['recherche_f']) == 0:
                st.subheader('Aucun match')
                len_df_f = False
            else:
                len_df_f = len(st.session_state['recherche_f'])
            
            ########################################################################## CONDITION D'AFFICHAGE DES FILMS FILTRES
            if len_df_f:
                real_len = str(len(st.session_state['recherche_f']))
                st.write('Resultats : ',str(int(real_len)))
                ########################################################################################### BOUCLE D'AFFICHAGE
                for i in range(int(real_len)):

                    if len(st.session_state['recherche_f']) == 1:
                        i = i

                    col_1, col_2, col_3= st.columns([1, 3, 1])
                    with col_1:
                        url_1 = jaquette(st.session_state.recherche_f.iloc[i]['poster_path'], url)
                        st.image(url_1, width=150)
                    with col_2:
                        st.markdown('***'+st.session_state.recherche_f.iloc[i]['Titre']+'***')
                    
                        nb_genre = len(st.session_state.recherche_f.iloc[i]['Genres'])
                        ############################################################################ CONDITION AFFICHAGE GENRES
                        if nb_genre == 1:
                            st.write("""
                            <style>
                            .tag {
                            font-size: 10px;
                            display: inline-block;
                            background-color: #ccc;
                            color: #000;
                            padding: 2px 5px;
                            border-radius: 15px;
                            margin-rght : 5px
                            }
                            </style>
                            <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][0],"""</span>
                            """, unsafe_allow_html=True)
                        
                        if nb_genre == 2:
                            st.write("""
                            <style>
                            .tag {
                            font-size: 10px;
                            display: inline-block;
                            background-color: #ccc;
                            color: #000;
                            padding: 2px 5px;
                            border-radius: 15px;
                            margin-rght : 5px
                            }
                            </style>
                            <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][0],"""</span>
                            <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][1],"""</span>
                            """, unsafe_allow_html=True)

                        if nb_genre == 3:
                            st.write("""
                            <style>
                            .tag {
                            font-size: 10px;
                            display: inline-block;
                            background-color: #ccc;
                            color: #000;
                            padding: 2px 5px;
                            border-radius: 15px;
                            margin-rght : 5px
                            }
                            </style>
                            <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][0],"""</span>
                            <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][1],"""</span>
                            <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][2],"""</span>
                            """, unsafe_allow_html=True)
                        ####################################################################################### RESUME DU FILMS
                        st.markdown('<p class="description">'+raccourcir_chaine(overview(st.session_state.recherche_f.iloc[i]['overview']))+'</p>', unsafe_allow_html=True)
                    with col_3:
                        ################################################################################# BOUTON IMDB ET RATING
                        st.write("""<a href=" """,url_get_imdb+st.session_state.recherche_f.iloc[i]['tconst'],""" " target="black_" style="box-shadow: 0px 1px 0px 0px #fff6af;background:linear-gradient(to bottom, #ffec64 5%, #ffab23 100%);background-color:#ffec64;border-radius:6px;border:1px solid #ffaa22;display:inline-block;cursor:pointer;color:#333333;font-family:Arial;font-size:12px;font-weight:bold;padding:6px 24px;text-decoration:none;text-shadow:0px 1px 0px #ffee66;" >Voir sur IMDB</a>""", unsafe_allow_html=True)
                        st.write("""<span style="font-size:50px; font-weight:bolder ">""",str(st.session_state.recherche_f.iloc[i]['Note_moyenne']),"""<sup>/10</sup></span>""", unsafe_allow_html=True)

    ##################################################################################### SOUMISSION PAR GENRE 
    elif reco_by == 'Par Genres':

        # RECUPERATION DES DONNEE ET FILTRE CHOIX 1
        r_genre = films_db[films_db['Genres'].apply(lambda x: any(genres in liked_genre for genres in x))]
        r_genre = r_genre.sort_values(['popularity', 'Note_moyenne', 'Nb_votes'], ascending=[False, False, False])

        # ATTRIBUTUION DES SESSIONS ( verifier que si elle existe on les ecrase )
        session['recherche'] = r_genre.head(50)
        session['recherche_f'] = r_genre.head(50)

        ######################################################################### CREATION DES VARIABLES FILTRES
        ss_unique_genres = st.session_state.recherche_f["Genres"].explode().unique()
        ss_unique_acteurs = st.session_state.recherche_f["liste_acteurs"].str.split(',').explode().unique()
        ss_unique_date = st.session_state.recherche_f["Annee"].unique()
        max_note = int(session.recherche['Note_moyenne'].max())
        ss_unique_note = [math.floor(i) for i in range(max_note + 1)]

        ########################################################################## AFFICHAGE FILTRES SECONDAIRES
        colx, colxx, colxxx, colxxxx = st.columns(4)
        with colx:
            ss_date = st.slider('Date',int(ss_unique_date.min()), int(ss_unique_date.max())+1, (int(ss_unique_date.min()), int(ss_unique_date.max())))
        with colxx:
            ss_genres = st.multiselect('Genres', ss_unique_genres, max_selections=3)
        with colxxx:
            ss_acteurs = st.multiselect('Acteurs', ss_unique_acteurs, max_selections=1 )
        with colxxxx:
            ss_note = st.selectbox('Note', ss_unique_note)

        ############################################################################### SI LES VARIABLES EXISTENT
        if ss_date or ss_genres or ss_acteurs or ss_note:
            # on filtre notre df de session avec la fonction 
            st.session_state['recherche_f'] = filtrer_films(st.session_state['recherche'], ss_genres, ss_acteurs, ss_date, ss_note)
            # on condition notre boucle
            if len(st.session_state['recherche_f']) >= 100:
                len_df_f = len(st.session_state['recherche_f'])-1
            elif len(st.session_state['recherche_f']) == 1:
                len_df_f = len(st.session_state['recherche_f'])
            elif len(st.session_state['recherche_f']) == 0:
                st.subheader('Aucun match')
                len_df_f = False
            else:
                len_df_f = len(st.session_state['recherche_f'])

            ############################################################################ AFFICHAGE DE LA RECHERCHE
            real_len = str(len(st.session_state['recherche_f']))
            st.write('Resultats : ', str(int(real_len)))

            ######################################################################### BOUCLE D'AFFICHAGE DES FILMS
            for i in range(int(real_len)):
                if(int(real_len)) == 1:
                    i=i
                col_1, col_2, col_3= st.columns([1, 3, 1])
                ###################################################################################### PATH POSTER
                with col_1:
                    url_1 = jaquette(st.session_state.recherche_f.iloc[i]['poster_path'], url)
                    st.image(url_1, width=150)
                with col_2:
                    ############################################################################## TITRE ET GENRES
                    st.markdown('***'+st.session_state.recherche_f.iloc[i]['Titre']+'***')
                    nb_genre = len(st.session_state.recherche_f.iloc[i]['Genres'])

                    ################################################################### CONDITION AFFICHAGE GENRES
                    if nb_genre == 1:
                        st.write("""
                        <style>
                        .tag {
                        font-size: 10px;
                        display: inline-block;
                        background-color: #ccc;
                        color: #000;
                        padding: 2px 5px;
                        border-radius: 15px;
                        margin-rght : 5px
                        }
                        </style>
                        <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][0],"""</span>
                        """, unsafe_allow_html=True)
                    
                    if nb_genre == 2:
                        st.write("""
                        <style>
                        .tag {
                        font-size: 10px;
                        display: inline-block;
                        background-color: #ccc;
                        color: #000;
                        padding: 2px 5px;
                        border-radius: 15px;
                        margin-rght : 5px
                        }
                        </style>
                        <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][0],"""</span>
                        <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][1],"""</span>
                        """, unsafe_allow_html=True)

                    if nb_genre == 3:
                        st.write("""
                        <style>
                        .tag {
                        font-size: 10px;
                        display: inline-block;
                        background-color: #ccc;
                        color: #000;
                        padding: 2px 5px;
                        border-radius: 15px;
                        margin-rght : 5px
                        }
                        </style>
                        <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][0],"""</span>
                        <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][1],"""</span>
                        <span class="tag">""",st.session_state.recherche_f.iloc[i]['Genres'][2],"""</span>
                        """, unsafe_allow_html=True)
                    ############################################################################### RESUME DU FILM
                    st.markdown('<p class="description">'+raccourcir_chaine(overview(st.session_state.recherche_f.iloc[i]['overview']))+'</p>', unsafe_allow_html=True)
                with col_3:
                    ######################################################################### BOUTON IMDB ET RATING
                    st.write("""<a href=" """,url_get_imdb+st.session_state.recherche_f.iloc[i]['tconst'],""" " target="black_" style="box-shadow: 0px 1px 0px 0px #fff6af;background:linear-gradient(to bottom, #ffec64 5%, #ffab23 100%);background-color:#ffec64;border-radius:6px;border:1px solid #ffaa22;display:inline-block;cursor:pointer;color:#333333;font-family:Arial;font-size:12px;font-weight:bold;padding:6px 24px;text-decoration:none;text-shadow:0px 1px 0px #ffee66;" >Voir sur IMDB</a>""", unsafe_allow_html=True)
                    st.write("""<span style="font-size:50px; font-weight:bolder ">""",str(st.session_state.recherche_f.iloc[i]['Note_moyenne']),"""<sup>/10</sup></span>""", unsafe_allow_html=True)
    else:
        ########################################################################### ERREUR SI CONDITION INTROUVABLE
        st.warning("Une erreur inconnu c'est produite.")

else:
    ############################################################################ RECUPERATION TOP 1
    top_1 = films_db.sort_values('popularity', ascending=False)[:1]
    #################################################################### RECUPERATION TOP 20 SAUF 1
    top_5_pop = films_db.sort_values('popularity', ascending=False)[1:21]

    #######################################################################RECUPERATION TOP 20 DRAMA
    drama_movies = films_db[films_db['Genres'].apply(lambda x: 'Drama' in x)]
    top_20_drama = drama_movies.sort_values('popularity', ascending=False)[:20]

    ##################################################################### RECUPERATION TOP 20 COMEDY
    comedy_movies = films_db[films_db['Genres'].apply(lambda x: 'Comedy' in x)]
    top_20_comedy = comedy_movies.sort_values('popularity', ascending=False)[:20]

    ################################################################### RECUPERATION TOP 20 THRILLER
    thriller_movies = films_db[films_db['Genres'].apply(lambda x: 'Thriller' in x)]
    top_20_thriller = thriller_movies.sort_values('popularity', ascending=False)[:20]      

    ####################################################################### RECUPERATION IMAGE TOP 20
    IMAGES_TOP = [
        jaquette(top_5_pop.iloc[0]['poster_path'], url),
        jaquette(top_5_pop.iloc[1]['poster_path'], url),
        jaquette(top_5_pop.iloc[2]['poster_path'], url),
        jaquette(top_5_pop.iloc[3]['poster_path'], url),
        jaquette(top_5_pop.iloc[4]['poster_path'], url),
        jaquette(top_5_pop.iloc[5]['poster_path'], url),
        jaquette(top_5_pop.iloc[6]['poster_path'], url),
        jaquette(top_5_pop.iloc[7]['poster_path'], url),
        jaquette(top_5_pop.iloc[8]['poster_path'], url),
        jaquette(top_5_pop.iloc[9]['poster_path'], url),
        jaquette(top_5_pop.iloc[10]['poster_path'], url),
        jaquette(top_5_pop.iloc[11]['poster_path'], url),
        jaquette(top_5_pop.iloc[12]['poster_path'], url),
        jaquette(top_5_pop.iloc[13]['poster_path'], url),
        jaquette(top_5_pop.iloc[14]['poster_path'], url),
        jaquette(top_5_pop.iloc[15]['poster_path'], url)

    ]
    ################################################################## RECUPERATION IMAGE TOP 20 DRAMA
    IMAGES_DRAMA = [
        jaquette(top_20_drama.iloc[0]['poster_path'], url),
        jaquette(top_20_drama.iloc[1]['poster_path'], url),
        jaquette(top_20_drama.iloc[2]['poster_path'], url),
        jaquette(top_20_drama.iloc[3]['poster_path'], url),
        jaquette(top_20_drama.iloc[4]['poster_path'], url),
        jaquette(top_20_drama.iloc[5]['poster_path'], url),
        jaquette(top_20_drama.iloc[6]['poster_path'], url),
        jaquette(top_20_drama.iloc[7]['poster_path'], url),
        jaquette(top_20_drama.iloc[8]['poster_path'], url),
        jaquette(top_20_drama.iloc[9]['poster_path'], url),
        jaquette(top_20_drama.iloc[10]['poster_path'], url),
        jaquette(top_20_drama.iloc[11]['poster_path'], url),
        jaquette(top_20_drama.iloc[12]['poster_path'], url),
        jaquette(top_20_drama.iloc[13]['poster_path'], url),
        jaquette(top_20_drama.iloc[14]['poster_path'], url),
        jaquette(top_20_drama.iloc[15]['poster_path'], url),

    ]
    ################################################################# RECUPERATION IMAGE TOP 20 COMEDY
    IMAGES_COMEDY= [
        jaquette(top_20_comedy.iloc[0]['poster_path'], url),
        jaquette(top_20_comedy.iloc[1]['poster_path'], url),
        jaquette(top_20_comedy.iloc[2]['poster_path'], url),
        jaquette(top_20_comedy.iloc[3]['poster_path'], url),
        jaquette(top_20_comedy.iloc[4]['poster_path'], url),
        jaquette(top_20_comedy.iloc[5]['poster_path'], url),
        jaquette(top_20_comedy.iloc[6]['poster_path'], url),
        jaquette(top_20_comedy.iloc[7]['poster_path'], url),
        jaquette(top_20_comedy.iloc[8]['poster_path'], url),
        jaquette(top_20_comedy.iloc[9]['poster_path'], url),
        jaquette(top_20_comedy.iloc[10]['poster_path'], url),
        jaquette(top_20_comedy.iloc[11]['poster_path'], url),
        jaquette(top_20_comedy.iloc[12]['poster_path'], url),
        jaquette(top_20_comedy.iloc[13]['poster_path'], url),
        jaquette(top_20_comedy.iloc[14]['poster_path'], url),
        jaquette(top_20_comedy.iloc[15]['poster_path'], url),

    ]
        # IMAGE POSTER TOP IMDB GEBNRE ACTION
    ##############################################################  RECUPERATION IMAGE TOP 20 THRILLER
    IMAGES_ACTION= [
        jaquette(top_20_thriller.iloc[0]['poster_path'], url),
        jaquette(top_20_thriller.iloc[1]['poster_path'], url),
        jaquette(top_20_thriller.iloc[2]['poster_path'], url),
        jaquette(top_20_thriller.iloc[3]['poster_path'], url),
        jaquette(top_20_thriller.iloc[4]['poster_path'], url),
        jaquette(top_20_thriller.iloc[5]['poster_path'], url),
        jaquette(top_20_thriller.iloc[6]['poster_path'], url),
        jaquette(top_20_thriller.iloc[7]['poster_path'], url),
        jaquette(top_20_thriller.iloc[8]['poster_path'], url),
        jaquette(top_20_thriller.iloc[9]['poster_path'], url),
        jaquette(top_20_thriller.iloc[10]['poster_path'], url),
        jaquette(top_20_thriller.iloc[11]['poster_path'], url),
        jaquette(top_20_thriller.iloc[12]['poster_path'], url),
        jaquette(top_20_thriller.iloc[13]['poster_path'], url),
        jaquette(top_20_thriller.iloc[14]['poster_path'], url),
        jaquette(top_20_thriller.iloc[15]['poster_path'], url),

    ]
    
    ##################################################################################### HEADER TOP 1 
    st.subheader('En ce moment ...')
    tab1, tab2 = st.tabs(["Le films", "Bande annonce"])
    with tab1:
        col1, col2 = st.columns([1,3])
        with col1:
            st.image(jaquette(top_1.iloc[0]['poster_path'], url))
        with col2:
            colx, coly = st.columns([3,1])
            with colx:
                st.header(top_1.iloc[0]['Titre'])
                st.write(top_1.iloc[0]['overview'])
                st.write("""<a href=" """,url_get_imdb+top_1.iloc[0]['tconst'],""" " target="black_" style="box-shadow: 0px 1px 0px 0px #fff6af;background:linear-gradient(to bottom, #ffec64 5%, #ffab23 100%);background-color:#ffec64;border-radius:6px;border:1px solid #ffaa22;display:inline-block;cursor:pointer;color:#333333;font-family:Arial;font-size:15px;font-weight:bold;padding:6px 24px;text-decoration:none;text-shadow:0px 1px 0px #ffee66;" >Voir sur IMDB</a>""", unsafe_allow_html=True)
            with coly:
                st.markdown("**CASTING**")
                listeslipacteurs = top_1.iloc[0]['liste_acteurs'].split(',')
                for acteurlist in range(len(listeslipacteurs)):
                    st.write(listeslipacteurs[acteurlist])
                st.markdown("**NOTE**")
                st.write("""<span style="font-size:50px; font-weight:bolder ">""",str(top_1.iloc[0]['Note_moyenne']),"""<sup>/10</sup></span>""", unsafe_allow_html=True)
    with tab2:
        st.video('https://imdb-video.media-imdb.com/vi3289957657/hls-preview-2ff398a0-3c6a-46d4-9caa-15583d5740e8.m3u8?Expires=1684348036&Signature=GakRwFxQCdBOT6EdC~UiuZJiCyurmWdotvsTWSBx0gDZYWVkWPCOlUhOxIPV5pVqyRrjx6NUXSlyhTDM5yTl7YLt57I3Vlql3ysoHk2ch41o7Oep~R6tNwOESgnWGporwC1FSFdeaJsTcbjclKuZLlyfpnBAoDpsS8eHVkmP7IcJVuRfjKv3IpAEUVGjbsHt8qmdriVHp9Fh5HxDYmOERnyDbJziBTswSWLSlJCLsVAgrOslqZKLna7TuS8mC2TbNz6EXPRjwd81pB8JkvIfud0czmBEYgCG-Wn1nis7oKtEAd1KkuWxbGo538mZnXDwCidxYLe3b7mt8JFgz1Iolw__&Key-Pair-Id=APKAIFLZBVQZ24NQH3KA')
        
    ################################################################### AFFICHAGE DU SLIDER TOP IMDB
    st.subheader("Top IMDB")
    slideshow_swipeable(IMAGES_TOP, top_5_pop)
    ################################################################## AFFICHAGE DU SLIDER TOP DRAMA
    st.subheader("Top des Drama")
    slideshow_swipeable(IMAGES_DRAMA, top_20_drama)
    ################################################################# AFFICHAGE DU SLIDER TOP COMEDY
    st.subheader("Top des Comedy")
    slideshow_swipeable(IMAGES_COMEDY, top_20_comedy)
    ############################################################### AFFICHAGE DU SLIDER TOP THRILLER
    st.subheader("Top des Thriller")
    slideshow_swipeable(IMAGES_ACTION, top_20_thriller)

    ################################################################## AFFICHAGE DU LOGO BAS DE PAGE
    st.markdown('---')
    st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown('<div class="centered"><img src="https://i.ibb.co/d5PHDkM/team.png" height="70"></div>', unsafe_allow_html=True)

