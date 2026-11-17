"""Board word pool: common, concrete-ish English nouns with multiple associations.

Self-made list (not the official Codenames list). Feel free to extend.
"""

_RAW = """
apple anchor angel ant arm army arrow bank bark bat battery beach bear bed bee bell
belt berry bill bird blade block board boat bolt bomb bond book boot bottle bow box
brain branch bread bridge brush bubble bucket bug button cake camera camp candle
cap capital car card castle cat cell chain chair chalk change check cheese chest
chicken chip church circle clock cloud club coach coat code coin cold comet compass
cook copper cotton court cover cow crane crash cross crown crystal cup current
diamond dice disk doctor dog dragon dream dress drill drop drum duck dust eagle
earth egg engine eye face fair fan farm feather fence field file fire fish flag
flame floor flute fly fog football forest fork fountain frog game garden gas ghost
giant glass glove gold grass ground gun hair hammer hand harbor hat head heart
helmet hero hole honey hook horn horse hospital hotel ice ink iron island jack jam
jet judge key king kite knife knight lab ladder lamp lead leaf lemon letter light
line lion lock log machine magic mail map marble mark mask match mercury milk mine
mint mirror model moon mouse mouth mud nail needle net night note nurse oil olive
opera orange organ palm pan paper park pass pen penguin piano pilot pin pipe pirate
pit plane plant plate play plot pocket point poison pole pool port post pound press
prince queen rabbit racket ray ring river robot rock rocket roll root rose round
ruler salt sand satellite school scale screen seal shadow shark ship shoe shop shot
silk sink skate sky slip snake snow soap sock soldier space spell spider spike spine
spring spy square stadium staff star steam steel stick stock storm straw stream
string sugar suit sun swan sword table tail tank tap teacher tea temple thread
throne tick tie tiger tin tooth torch tower track train tree triangle truck trunk
tube turkey umbrella van violin volcano wall war watch water wave web whale wheel
whip wind window wine witch wolf wood worm yard
"""

_EXTRA = """
alien alarm almond amber arena atlas avalanche badge balloon bamboo bandage barrel
basket beard blanket blizzard bubble cactus canal cannon canvas carpet carrot cave
cherry cliff cloak comb concert crater cricket curtain desert dinosaur dolphin
drone engine envelope feast ferry flower galaxy glacier globe goblin grape guitar
hammock harvest hive igloo jacket jungle kettle ladle laser lighthouse magnet
marathon meteor microscope monkey mummy museum ninja ocean orbit owl parrot
passport peanut pepper pillow planet puzzle pyramid radar rainbow saddle sailor
scarf scorpion shell skull sleeve spoon statue sunflower telescope thunder ticket
tornado tractor treasure trumpet tunnel vampire vault wallet wizard zombie
acorn admiral airport alley antenna apron archer armor attic avocado bagel ballet
bandit banner barber beacon beetle bishop blender bonfire border bracelet breeze
broom buffalo butler cabin camel candy captain castle cereal chess chimney chorus
circus clown cobra coffee comet coral costume cowboy crayon cyclone dagger detective
diary diploma dragon eclipse elephant emerald factory falcon fiddle fossil furnace
gallery garlic gazelle geyser gorilla harbor hurricane jaguar jester journal
kangaroo keyboard kingdom lantern lizard llama locket lobster mammoth mansion
meadow medal melon mermaid miner monster mosquito necklace noodle oasis octopus
orchestra oyster paddle palace panda parade peacock pearl pebble pencil phantom
pickle pilgrim pirate plumber poet potato prism puppet python quarry raven reactor
rhythm saloon scarecrow sheriff shield siren skeleton sleigh slipper spiral sponge
squirrel stamp stove submarine summit surgeon swamp tablet theater tiger toast
toilet trophy turtle unicorn valley velvet viking vulture waffle walrus whistle
"""

MIN_LEN = 5  # short words (cat, moon, ...) are excluded from the board

WORDS: list[str] = sorted({w for w in (_RAW + _EXTRA).split() if len(w) >= MIN_LEN})
