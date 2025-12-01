import os
from datetime import datetime, date
from dateutil.parser import parse as parse_date
from typing import List, Dict, Any, Optional

import streamlit as st
import google.generativeai as genai


# =========================
# 1. CONFIG & GEMINI-CLIENT
# =========================

# Zet in Streamlit Cloud bij Settings -> Secrets:
# GEMINI_API_KEY = "jouw_api_key"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-1.5-pro"


# =========================
# 2. SYSTEM PROMPT (UIT JOUW PDF)
# =========================

SYSTEM_PROMPT = """
Je bent een AI-assistent in een interne web-app voor Land Automotive B.V. (B2B-autohandelaar).
Je werkt ALTIJD in het Nederlands.

DOEL VAN HET SYSTEEM
- De gebruiker sleept een inkoopfactuur (PDF/scan) of plakt OCR-tekst.
- Jij leest alle relevante gegevens uit.
- Jij helpt met:
  1) Mobilox-gegevens / voertuigdata
  2) Locatie + mailtekst voor transporteur
  3) Takenlijst per auto
  4) Inspectierapport + (optionele) kostenraming
  5) Labeltekst voor sleutellabel
- Je verzint geen gegevens: als iets niet in de bron staat, laat je het leeg of gebruik je null.

ALGEMENE REGELS
- Antwoord ALTIJD in dezelfde vaste blokken.
- Geef eerst een KORTE samenvatting voor mensen.
- Daarna gestructureerde blokken die de app kan gebruiken (bijvoorbeeld JSON-achtige structuren).
- Verzin GEEN kenteken / chassisnummer / bedragen als deze niet in de bron voorkomen.

BLOK 1 – MOBILOX-GEGEVENS & INTERNE VOERTUIGDATA
Je probeert minimaal deze velden te bepalen (indien beschikbaar in tekst/factuur):

- leverancier_naam
- leverancier_plaats
- factuurnummer
- factuurdatum
- merk
- model
- type_of_uitvoering
- brandstof            (benzine, diesel, hybride, PHEV, EV)
- carrosserie          (hatchback, station, SUV, etc.)
- kenteken
- meldcode             (laatste 4 cijfers chassisnummer als Nederlands kenteken)
- chassisnummer
- bouwjaar
- datum_eerste_toelating
- kilometerstand
- kleur
- transmissie
- vermogen_pk
- vermogen_kw
- btw_of_marge_auto    ("BTW" of "Marge")
- inkoopprijs_excl_btw
- btw_bedrag
- inkoopprijs_incl_btw
- bpm_bedrag_op_factuur (indien op factuur vermeld)

OUTPUT:
1) "Samenvatting" – 1–3 regels tekst.
2) "Mobilox-gegevens" – velden in overzichtelijke vorm (geschikt om te kopiëren).
3) "Interne voertuigdata" – zelfde info maar technisch gestructureerd (bijv. JSON-achtig).

BLOK 2 – LOCATIE & E-MAIL NAAR TRANSPORTEUR
- Bepaal waar de auto nu staat (standplaats).
- Velden (indien mogelijk uit tekst/factuur):

  standplaats_naam
  standplaats_adres
  standplaats_postcode
  standplaats_plaats
  standplaats_land
  contactpersoon_naam
  contactpersoon_telefoon
  openingstijden

- Genereer een korte, zakelijke mail voor de transporteur.
- Let op: Ophaallocatie is NOOIT Land Automotive zelf.
- Leveradres is standaard: "Land Automotive B.V., Veen" (de app vult het later precies in).

OUTPUT:
4) "Locatie voertuig" – velden.
5) "E-mail aan transporteur (concept)" – onderwerp + tekst.

BLOK 3 – TAKEN PER AUTO
Takenlogica:

Altijd:
- transport_plannen
- online_zetten
- check_extra_werk

Extra:
- brandstofauto (benzine/diesel, géén PHEV/EV): taak "taxatie_inplannen"
- PHEV: taak "bpm_rapport_maken"
- EV: geen BPM-rapport, eventueel aanvullende taak "actieradius_testen"

Per taak geef je:
- taak_id (korte technische naam, bijv. "transport_plannen")
- taak_naam (leesbare naam)
- omschrijving
- prioriteit (hoog / midden / laag)
- status ("open")
- categorie (bijv. logistiek, administratie, verkoopvoorbereiding, techniek)
- automatisch_gegenereerd = true

OUTPUT:
6) "Taken voor deze auto" – lijst van taken (gestructureerd, bijv. JSON-array) + korte opsomming.

BLOK 4 – INSPECTIE (OPTIONEEL)
Als de gebruiker inspectie-info aanlevert (tekst, schadebeschrijving), maak je:

- "Inspectie-data (gestructureerd)" met o.a.:
  - rubrieken (Exterieur, Interieur, Banden/velgen, Ruiten/verlichting, Technisch, Elektronica)
  - per schade: locatie, beschrijving, schade_type, schatting_uren, schatting_materiaal_kosten

- "Inspectierapport – tekst voor PDF"
  - heel zakelijk, zonder uitgebreide proza
  - bovenaan: Land Automotive B.V., datum, merk, model, kleur, kenteken (als NL), chassisnummer, km-stand, datum deel 1.
  - toon verkoopfoto’s bovenaan ZONDER tekst
  - toon schadefoto’s onderaan MET korte omschrijving
  - als er kostenraming is: geef alleen tabel of opsomming (geen lange verhalende tekst)

Als er geen inspectie-informatie is, vermeld je duidelijk dat de inspectieblokken niet worden gevuld.

BLOK 5 – LABEL VOOR SLEUTELLABEL
Doel: kort label voor op sleutel, max 4 regels.

- Standaard:
  Regel 1: Land Automotive
  Regel 2: [merk] [model] [type/uitvoering]
  Regel 3: [kenteken] (als NL) OF laatste 4 tekens van chassisnummer
  Regel 4: kleur + (indien relevant) PHEV / EV / Diesel

Output:
7) "Label voor sleutellabel" met:
   - regel_1, regel_2, regel_3, regel_4
   - label_samengevat (1 korte regel)

ALGEMENE OUTPUTVOLGORDE
Producing ALTIJD (voor zover van toepassing):

- Samenvatting
- Mobilox-gegevens
- Interne voertuigdata
- Locatie voertuig
- E-mail aan transporteur (concept)
- Taken voor deze auto
- Inspectie-data (gestructureerd) – alleen als info aanwezig
- Inspectierapport – tekst voor PDF – alleen als info aanwezig
- Label voor sleutellabel

Gebruik duidelijke kopjes per blok.
Zet velden netjes onder elkaar.
"""

# =========================
# 3. HULPFUNCTIES: AI & DATA
# =========================

def call_gemini(system_prompt: str, user_content: str) -> str:
    """Roep Gemini aan met een system-instructie + user-tekst."""
    if not GEMINI_API_KEY:
        return "⚠️ Geen GEMINI_API_KEY ingesteld. Zet deze in de Streamlit Secrets."

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            user_content,
        )
        return response.text or ""
    except Exception as e:
        # Toon de fouttekst in de app zodat we weten wat er misgaat
        return f"⚠️ Fout bij aanroepen van Gemini: {e}"


def init_state():
    """Initialiseer alle state-structuren één keer."""
    if "cars" not in st.session_state:
        st.session_state.cars: List[Dict[str, Any]] = []
    if "customers" not in st.session_state:
        st.session_state.customers: List[Dict[str, Any]] = []
    if "transporters" not in st.session_state:
        st.session_state.transporters: List[Dict[str, Any]] = []
    if "suppliers" not in st.session_state:
        st.session_state.suppliers: List[Dict[str, Any]] = []
    if "invoices" not in st.session_state:
        st.session_state.invoices: List[Dict[str, Any]] = []
    if "backup_json" not in st.session_state:
        st.session_state.backup_json = ""


def new_car_id() -> str:
    return f"car_{int(datetime.utcnow().timestamp()*1000)}"


def compute_stand_days(created_at: date) -> int:
    return (date.today() - created_at).days


# =========================
# 4. PAGINA'S
# =========================

def page_dashboard():
    st.header("Dashboard – Overzicht voertuigen")

    if not st.session_state.cars:
        st.info("Er zijn nog geen voertuigen. Kies in de sidebar 'Nieuwe auto' om te starten.")
        return

    search = st.text_input("Zoek op merk, model, kenteken of chassisnummer")
    filtered = []
    for car in st.session_state.cars:
        text = " ".join(
            str(car.get("vehicle_data", {}).get(k, "")) for k in
            ["merk", "model", "kenteken", "chassisnummer"]
        ).lower()
        if not search or search.lower() in text:
            filtered.append(car)

    for car in filtered:
        vd = car.get("vehicle_data", {})
        merk = vd.get("merk", "Onbekend")
        model = vd.get("model", "")
        kenteken = vd.get("kenteken", "")
        status = car.get("status", "Te koop")
        verkoopprijs = vd.get("verkoopprijs_incl_btw", None)
        created_at = car.get("created_at_date", date.today())
        stadagen = compute_stand_days(created_at)

        with st.container(border=True):
            st.subheader(f"{merk} {model} {('– '+kenteken) if kenteken else ''}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Status:** {status}")
                st.write(f"**Stadagen:** {stadagen}")
            with col2:
                if verkoopprijs is not None:
                    st.write(f"**Verkoopprijs (incl. BTW):** € {verkoopprijs:,.0f}".replace(",", "."))
            with col3:
                if st.button("Open dossier", key=f"open_{car['id']}"):
                    st.session_state["active_car_id"] = car["id"]
                    st.session_state["active_page"] = "Dossier"


def page_new_car():
    st.header("Nieuwe auto – Inkoopfactuur inlezen")

    st.write("Sleep hier je inkoopfactuur (PDF/JPG/PNG) of plak tekst.")
    uploaded = st.file_uploader("Inkoopfactuur (PDF/beeld)", type=["pdf", "jpg", "jpeg", "png"])
    ocr_text = st.text_area("Of plak hier de tekst van de factuur/advertentie", height=200)
    extra_context = st.text_area(
        "Extra info (bijv. waar de auto staat, bijzonderheden, klant, interne notities)",
        height=120,
    )

    if st.button("Verwerk met AI"):
        if not uploaded and not ocr_text:
            st.warning("Upload een factuur of plak tekst voordat je op 'Verwerk met AI' klikt.")
            return

        with st.spinner("Factuur wordt verwerkt door AI…"):
            # In deze eerste versie gebruiken we alleen de tekst.
            # Later kun je hier OCR toevoegen voor PDF/beelden.
            base_text = ocr_text or ""
            if uploaded:
                base_text += f"\n[BESTANDSNAAM: {uploaded.name}]"

            ai_output = call_gemini(SYSTEM_PROMPT, base_text)

        # Sla ruwe AI-output op in de car – de app kan dit later verder parsen/splitsen.
        car = {
            "id": new_car_id(),
            "created_at": datetime.utcnow().isoformat(),
            "created_at_date": date.today(),
            "raw_ai_output": ai_output,
            "vehicle_data": {},   # later te vullen uit ai_output
            "tasks": [],
            "costs": [],
            "notes": extra_context,
            "status": "Te koop",
        }
        st.session_state.cars.append(car)
        st.session_state["active_car_id"] = car["id"]
        st.session_state["active_page"] = "Dossier"

        st.success("Auto aangemaakt op basis van de factuur. Ga verder in het dossier.")
        st.markdown("### AI-output (debug / controle)")
        st.code(ai_output)


def get_active_car() -> Optional[Dict[str, Any]]:
    car_id = st.session_state.get("active_car_id")
    if not car_id:
        return None
    for c in st.session_state.cars:
        if c["id"] == car_id:
            return c
    return None


def page_dossier():
    car = get_active_car()
    if not car:
        st.info("Geen actief dossier. Ga naar het Dashboard en kies een voertuig.")
        return

    vd = car.setdefault("vehicle_data", {})

    st.header("Voertuigdossier")

    tabs = st.tabs(["Gegevens", "Taken", "Kosten", "Inspectie", "Label", "Factuur (basis)"])

    # ---- Gegevens-tab ----
    with tabs[0]:
        st.subheader("Gegevens – voertuig & financieel")

        col1, col2, col3 = st.columns(3)

        with col1:
            vd["merk"] = st.text_input("Merk", vd.get("merk", ""))
            vd["model"] = st.text_input("Model", vd.get("model", ""))
            vd["type_of_uitvoering"] = st.text_input("Type/uitvoering", vd.get("type_of_uitvoering", ""))
            vd["kenteken"] = st.text_input("Kenteken", vd.get("kenteken", ""))
            vd["chassisnummer"] = st.text_input("Chassisnummer (VIN)", vd.get("chassisnummer", ""))

        with col2:
            vd["brandstof"] = st.text_input("Brandstof", vd.get("brandstof", ""))
            vd["transmissie"] = st.text_input("Transmissie", vd.get("transmissie", ""))
            vd["kleur"] = st.text_input("Kleur", vd.get("kleur", ""))
            vd["meldcode"] = st.text_input("Meldcode", vd.get("meldcode", ""))
            datum_1 = st.text_input("Datum deel 1", vd.get("datum_eerste_toelating", ""))

        with col3:
            vd["inkoopprijs_excl_btw"] = st.number_input(
                "Inkoopprijs excl. BTW",
                value=float(vd.get("inkoopprijs_excl_btw", 0) or 0),
                step=100.0,
            )
            vd["bpm_bedrag"] = st.number_input(
                "BPM bedrag",
                value=float(vd.get("bpm_bedrag", 0) or 0),
                step=100.0,
            )
            vd["verkoopprijs_incl_btw"] = st.number_input(
                "Verkoopprijs incl. BTW",
                value=float(vd.get("verkoopprijs_incl_btw", 0) or 0),
                step=100.0,
            )
            vd["btw_of_marge_auto"] = st.selectbox(
                "BTW of Marge auto",
                ["Onbekend", "BTW", "Marge"],
                index=["Onbekend", "BTW", "Marge"].index(vd.get("btw_of_marge_auto", "Onbekend"))
            )

        if datum_1:
            vd["datum_eerste_toelating"] = datum_1

        st.success("Gegevens worden automatisch in het dossier opgeslagen.")

    # ---- Taken-tab ----
    with tabs[1]:
        st.subheader("Taken")

        tasks: List[Dict[str, Any]] = car.setdefault("tasks", [])

        if not tasks:
            st.info("Nog geen taken gegenereerd. Klik hieronder om op basis van voertuiggegevens taken aan te maken.")
            if st.button("Genereer standaard taken"):
                brandstof = vd.get("brandstof", "").lower()
                new_tasks = []

                def add_task(id_, naam, omschrijving, categorie, prioriteit="midden"):
                    new_tasks.append({
                        "taak_id": id_,
                        "taak_naam": naam,
                        "omschrijving": omschrijving,
                        "categorie": categorie,
                        "prioriteit": prioriteit,
                        "status": "open",
                        "automatisch_gegenereerd": True,
                    })

                # Altijd
                add_task("transport_plannen", "Transport plannen", "Plan transport vanaf leverancier naar Land Automotive.", "logistiek", "hoog")
                add_task("online_zetten", "Online zetten", "Zet het voertuig online in Mobilox/website.", "verkoopvoorbereiding", "hoog")
                add_task("check_extra_werk", "Extra werk checken", "Controleer cosmetisch/technisch werk en poetsen.", "techniek", "midden")

                # Brandstof-specifiek
                if "phev" in brandstof or "plug" in brandstof:
                    add_task("bpm_rapport_maken", "BPM-rapport maken", "Maak een BPM-rapport voor deze PHEV.", "administratie", "midden")
                elif "ev" in brandstof or "elektrisch" in brandstof:
                    add_task("actieradius_testen", "Actieradius testen", "Test globaal de actieradius van de EV.", "techniek", "laag")
                elif "benzine" in brandstof or "diesel" in brandstof:
                    add_task("taxatie_inplannen", "Taxatie inplannen", "Plan een taxatie in voor deze brandstofauto.", "administratie", "midden")

                car["tasks"] = new_tasks
                st.experimental_rerun()
        else:
            for i, t in enumerate(tasks):
                cols = st.columns([3, 2, 2, 2])
                with cols[0]:
                    st.write(f"**{t['taak_naam']}**")
                    st.caption(t["omschrijving"])
                with cols[1]:
                    t["status"] = st.selectbox(
                        "Status",
                        ["open", "bezig", "afgerond"],
                        index=["open", "bezig", "afgerond"].index(t.get("status", "open")),
                        key=f"status_{car['id']}_{i}",
                    )
                with cols[2]:
                    t["prioriteit"] = st.selectbox(
                        "Prioriteit",
                        ["hoog", "midden", "laag"],
                        index=["hoog", "midden", "laag"].index(t.get("prioriteit", "midden")),
                        key=f"prio_{car['id']}_{i}",
                    )
                with cols[3]:
                    st.write(t.get("categorie", ""))

    # ---- Kosten-tab (basis) ----
    with tabs[2]:
        st.subheader("Kosten & resultaat (basis)")

        costs: List[Dict[str, Any]] = car.setdefault("costs", [])

        with st.form("add_cost"):
            col1, col2, col3 = st.columns(3)
            with col1:
                oms = st.text_input("Omschrijving", "")
            with col2:
                bedrag = st.number_input("Bedrag", min_value=0.0, step=50.0)
            with col3:
                incl_btw = st.selectbox("Bedrag incl. of excl. BTW?", ["incl", "excl"])
            submitted = st.form_submit_button("Kostenregel toevoegen")

        if submitted and oms and bedrag > 0:
            costs.append({
                "omschrijving": oms,
                "bedrag": bedrag,
                "incl_of_excl": incl_btw,
            })
            st.success("Kostenregel toegevoegd.")
            st.experimental_rerun()

        if costs:
            st.write("### Kosten")
            for c in costs:
                st.write(f"- {c['omschrijving']}: € {c['bedrag']:,.0f} ({c['incl_of_excl']})".replace(",", "."))

        inkoop = float(vd.get("inkoopprijs_excl_btw", 0) or 0)
        bpm = float(vd.get("bpm_bedrag", 0) or 0)
        verkoop_incl = float(vd.get("verkoopprijs_incl_btw", 0) or 0)

        # Uitgaan van 21% BTW indien BTW-auto
        btw_auto = vd.get("btw_of_marge_auto") == "BTW"
        if btw_auto and verkoop_incl:
            verkoop_excl = verkoop_incl / 1.21
        else:
            verkoop_excl = verkoop_incl  # bij marge of onbekend, simpel benaderd

        totale_kosten = sum(c["bedrag"] for c in costs)
        resultaat = verkoop_excl - inkoop - bpm - totale_kosten

        st.markdown("---")
        st.write(f"**Inkoop excl. BTW:** € {inkoop:,.0f}".replace(",", "."))
        st.write(f"**BPM:** € {bpm:,.0f}".replace(",", "."))
        st.write(f"**Totale kosten (extra):** € {totale_kosten:,.0f}".replace(",", "."))
        st.write(f"**Verkoop (netto benadering):** € {verkoop_excl:,.0f}".replace(",", "."))
        st.success(f"**Indicatief resultaat:** € {resultaat:,.0f}".replace(",", "."))

    # ---- Inspectie-tab (placeholder) ----
    with tabs[3]:
        st.subheader("Inspectie (basisversie)")
        st.info("Hier kun je later inspectietekst en foto's toevoegen. De AI kan dan een inspectierapport maken.")
        inspectietekst = st.text_area("Inspectie / schades (tekst)", car.get("inspectietekst", ""), height=200)
        if st.button("Genereer inspectierapport (AI)"):
            with st.spinner("Rapport wordt gemaakt…"):
                tekst = f"INSPECTIE-INFORMATIE:\n{inspectietekst}\n\nAUTO:\n{vd}"
                insp_output = call_gemini(SYSTEM_PROMPT, tekst)
            car["inspectierapport_ai"] = insp_output
            st.success("Inspectierapport gegenereerd.")
            st.markdown("### Inspectierapport (AI)")
            st.code(insp_output)
        car["inspectietekst"] = inspectietekst

        if "inspectierapport_ai" in car:
            st.markdown("### Laatste inspectierapport")
            st.code(car["inspectierapport_ai"])

    # ---- Label-tab ----
    with tabs[4]:
        st.subheader("Label voor sleutellabel")

        merk = vd.get("merk", "")
        model = vd.get("model", "")
        uitvoer = vd.get("type_of_uitvoering", "")
        kenteken = vd.get("kenteken", "")
        chassis = vd.get("chassisnummer", "")
        kleur = vd.get("kleur", "")
        brandstof = vd.get("brandstof", "")

        # Kenteken of laatste 4 van VIN
        if kenteken:
            regel3 = kenteken
        else:
            regel3 = chassis[-4:] if chassis else ""

        regel1 = "Land Automotive"
        regel2 = f"{merk} {model} {uitvoer}".strip()
        regel4 = f"{kleur} {(' '+brandstof) if brandstof else ''}".strip()

        st.text_input("Regel 1", value=regel1)
        st.text_input("Regel 2", value=regel2)
        st.text_input("Regel 3", value=regel3)
        st.text_input("Regel 4", value=regel4)

        st.info("In een latere versie kunnen we hier een echte printlayout / PDF voor een labelprinter van maken.")

    # ---- Factuur-tab (basis) ----
    with tabs[5]:
        st.subheader("Factuur (basisconcept)")

        st.write("Hier komt later de volledige factuurmodule (PDF, mail, credits, losse facturen, etc.).")
        st.write("Voor nu kun je de verkoopprijs en klantgegevens gebruiken om handmatig in je boekhoudpakket te factureren.")


def page_invoices():
    st.header("Facturen (overzicht)")
    st.info("In deze eerste versie is de factuurmodule nog niet volledig uitgewerkt. Dit wordt een aparte stap.")


def page_customers():
    st.header("Klanten (basis CRM)")

    customers = st.session_state.customers

    with st.form("add_customer"):
        st.subheader("Nieuwe klant toevoegen")
        naam = st.text_input("Bedrijfsnaam / klantnaam")
        email = st.text_input("E-mailadres")
        plaats = st.text_input("Plaats")
        submit = st.form_submit_button("Opslaan")
    if submit and naam:
        customers.append({"naam": naam, "email": email, "plaats": plaats})
        st.success("Klant toegevoegd.")

    if customers:
        st.markdown("### Bestaande klanten")
        for c in customers:
            st.write(f"- **{c['naam']}** ({c.get('plaats','')}) – {c.get('email','')}")


def page_relations():
    st.header("Relaties – Transporteurs & Leveranciers")

    transporters = st.session_state.transporters
    suppliers = st.session_state.suppliers

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transporteurs")
        with st.form("add_transporter"):
            naam = st.text_input("Naam transporteur")
            email = st.text_input("E-mailadres transporteur")
            tel = st.text_input("Telefoonnummer")
            submit = st.form_submit_button("Opslaan")
        if submit and naam:
            transporters.append({"naam": naam, "email": email, "telefoon": tel})
            st.success("Transporteur toegevoegd.")

        if transporters:
            st.write("### Bestaande transporteurs")
            for t in transporters:
                st.write(f"- **{t['naam']}** – {t.get('email','')} – {t.get('telefoon','')}")

    with col2:
        st.subheader("Leveranciers")
        with st.form("add_supplier"):
            naam = st.text_input("Naam leverancier", key="sup_naam")
            email = st.text_input("E-mailadres leverancier", key="sup_email")
            tel = st.text_input("Telefoonnummer leverancier", key="sup_tel")
            submit2 = st.form_submit_button("Opslaan", key="sup_submit")
        if submit2 and naam:
            suppliers.append({"naam": naam, "email": email, "telefoon": tel})
            st.success("Leverancier toegevoegd.")

        if suppliers:
            st.write("### Bestaande leveranciers")
            for s in suppliers:
                st.write(f"- **{s['naam']}** – {s.get('email','')} – {s.get('telefoon','')}")


def page_settings():
    st.header("Instellingen & backup")

    st.subheader("Backup maken")
    if st.button("Genereer JSON-backup"):
        data = {
            "cars": st.session_state.cars,
            "customers": st.session_state.customers,
            "transporters": st.session_state.transporters,
            "suppliers": st.session_state.suppliers,
            "invoices": st.session_state.invoices,
        }
        st.session_state.backup_json = str(data)
        st.download_button(
            "Download backup.json",
            data=st.session_state.backup_json,
            file_name="land_automotive_backup.json",
            mime="application/json",
        )

    st.subheader("Backup terugzetten")
    uploaded = st.file_uploader("Upload eerder gedownloade backup.json", type=["json"], key="backup_file")
    if uploaded and st.button("Backup inladen"):
        try:
            text = uploaded.read().decode("utf-8")
            # ALLES is string -> eval is niet ideaal, maar voor intern gebruik kan dit;
            # in productie beter json gebruiken met json.loads.
            data = eval(text)
            st.session_state.cars = data.get("cars", [])
            st.session_state.customers = data.get("customers", [])
            st.session_state.transporters = data.get("transporters", [])
            st.session_state.suppliers = data.get("suppliers", [])
            st.session_state.invoices = data.get("invoices", [])
            st.success("Backup succesvol teruggezet.")
        except Exception as e:
            st.error(f"Kon backup niet inladen: {e}")


# =========================
# 5. MAIN
# =========================

def main():
    # Basisconfig
    st.set_page_config(page_title="Land Automotive – Inkoop & Dossier", layout="wide")
    init_state()

    # Sidebar
    st.sidebar.title("Land Automotive")

    # Zorg dat active_page altijd bestaat
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Dashboard"

    # Menu in de sidebar
    page = st.sidebar.radio(
        "Menu",
        ["Dashboard", "Nieuwe auto", "Dossier", "Facturen", "Klanten", "Relaties", "Instellingen"],
        index=["Dashboard", "Nieuwe auto", "Dossier", "Facturen", "Klanten", "Relaties", "Instellingen"].index(
            st.session_state["active_page"]
        ),
    )

    # Routering naar de juiste pagina
    if page == "Dashboard":
        st.session_state["active_page"] = "Dashboard"
        page_dashboard()
    elif page == "Nieuwe auto":
        st.session_state["active_page"] = "Nieuwe auto"
        page_new_car()
    elif page == "Dossier":
        st.session_state["active_page"] = "Dossier"
        page_dossier()
    elif page == "Facturen":
        st.session_state["active_page"] = "Facturen"
        page_invoices()
    elif page == "Klanten":
        st.session_state["active_page"] = "Klanten"
        page_customers()
    elif page == "Relaties":
        st.session_state["active_page"] = "Relaties"
        page_relations()
    elif page == "Instellingen":
        st.session_state["active_page"] = "Instellingen"
        page_settings()


if __name__ == "__main__":
    main()
