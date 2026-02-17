import { useState, useMemo } from 'react'
import {
  Paper,
  TextInput,
  Group,
  Badge,
  Text,
  Progress,
  Stack,
  Box,
  Modal,
  ScrollArea,
  SimpleGrid,
  RingProgress,
  Center
} from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { DataTable, DataTableSortStatus } from 'mantine-datatable'
import { IconSearch, IconVideo, IconEye, IconPlayerPlay } from '@tabler/icons-react'
import { ActionIcon, Tooltip } from '@mantine/core'
import { VideoPreview } from './VideoPreview'
import type { VideoData } from '../types'

const PAGE_SIZE = 15

const EMOTION_COLORS: Record<string, string> = {
  happy: '#40c057',
  sad: '#748ffc',
  angry: '#ff6b6b',
  fear: '#f59f00',
  surprise: '#da77f2',
  disgust: '#20c997',
  neutral: '#868e96'
}

interface VideosTableProps {
  videos: VideoData[]
}

export function VideosTable({ videos }: VideosTableProps) {
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [sortStatus, setSortStatus] = useState<DataTableSortStatus<VideoData>>({
    columnAccessor: 'name',
    direction: 'asc'
  })
  const [selectedVideo, setSelectedVideo] = useState<VideoData | null>(null)
  const [detailsOpened, { open: openDetails, close: closeDetails }] = useDisclosure(false)
  const [videoPreviewName, setVideoPreviewName] = useState<string | null>(null)
  const [videoPreviewOpened, { open: openVideoPreview, close: closeVideoPreview }] = useDisclosure(false)

  // Filter and sort
  const filteredData = useMemo(() => {
    let data = [...videos]
    
    if (search) {
      const searchLower = search.toLowerCase()
      data = data.filter(v => v.name.toLowerCase().includes(searchLower))
    }
    
    const { columnAccessor, direction } = sortStatus
    data.sort((a, b) => {
      const aVal = a[columnAccessor as keyof VideoData]
      const bVal = b[columnAccessor as keyof VideoData]
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return direction === 'asc' ? aVal - bVal : bVal - aVal
      }
      
      return direction === 'asc'
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal))
    })
    
    return data
  }, [videos, search, sortStatus])

  const paginatedData = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE
    return filteredData.slice(start, start + PAGE_SIZE)
  }, [filteredData, page])

  const handleRowClick = (video: VideoData) => {
    setSelectedVideo(video)
    openDetails()
  }

  const handleVideoPreview = (videoName: string) => {
    setVideoPreviewName(videoName)
    openVideoPreview()
  }

  // Get top emotion for a video
  const getTopEmotion = (breakdown: Record<string, number>) => {
    const entries = Object.entries(breakdown)
    if (entries.length === 0) return { emotion: 'unknown', count: 0 }
    return entries.reduce((a, b) => a[1] > b[1] ? a : b, ['unknown', 0])
      .reduce((_, curr, i, arr) => ({ emotion: arr[0] as string, count: arr[1] as number }), { emotion: 'unknown', count: 0 })
  }

  return (
    <>
      <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        <Group mb="md" gap="sm">
          <TextInput
            placeholder="Search videos..."
            leftSection={<IconSearch size={16} />}
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
            style={{ flex: 1, maxWidth: 300 }}
          />
        </Group>

        <Text size="sm" c="dimmed" mb="sm">
          {filteredData.length} videos
        </Text>

        <DataTable
          withTableBorder={false}
          borderRadius="md"
          striped
          highlightOnHover
          records={paginatedData}
          columns={[
            {
              accessor: 'name',
              title: 'Video Name',
              sortable: true,
              width: 200,
              render: (video) => (
                <Group gap="xs">
                  <IconVideo size={16} color="#3395f3" />
                  <Text size="sm" fw={500} truncate>
                    {video.name}
                  </Text>
                </Group>
              )
            },
            {
              accessor: 'totalScenes',
              title: 'Total Scenes',
              sortable: true,
              width: 110,
              textAlign: 'center',
              render: (video) => (
                <Badge variant="light" color="cerulean">
                  {video.totalScenes}
                </Badge>
              )
            },
            {
              accessor: 'scenesWithPerson',
              title: 'With Person',
              sortable: true,
              width: 120,
              render: (video) => (
                <Group gap="xs">
                  <Progress
                    value={(video.scenesWithPerson / video.totalScenes) * 100}
                    color="green"
                    size="sm"
                    style={{ flex: 1 }}
                  />
                  <Text size="xs" c="dimmed" w={40}>
                    {video.scenesWithPerson}
                  </Text>
                </Group>
              )
            },
            {
              accessor: 'avgDetectionCoverage',
              title: 'Avg Coverage',
              sortable: true,
              width: 120,
              render: (video) => (
                <Progress
                  value={video.avgDetectionCoverage * 100}
                  color={video.avgDetectionCoverage > 0.6 ? 'green' : video.avgDetectionCoverage > 0.3 ? 'yellow' : 'gray'}
                  size="sm"
                />
              )
            },
            {
              accessor: 'emotionBreakdown',
              title: 'Top Emotion',
              width: 120,
              render: (video) => {
                const top = Object.entries(video.emotionBreakdown)
                  .sort(([,a], [,b]) => b - a)[0]
                return top ? (
                  <Badge variant="light" color={EMOTION_COLORS[top[0]] ? top[0] : 'gray'}>
                    {top[0]} ({top[1]})
                  </Badge>
                ) : (
                  <Text size="sm" c="dimmed">—</Text>
                )
              }
            },
            {
              accessor: 'avgValence',
              title: 'Avg Valence',
              sortable: true,
              width: 100,
              render: (video) => (
                <Text
                  size="sm"
                  c={video.avgValence > 0 ? 'green' : video.avgValence < -0.3 ? 'red' : 'dimmed'}
                  fw={500}
                >
                  {video.avgValence.toFixed(2)}
                </Text>
              )
            },
            {
              accessor: 'avgArousal',
              title: 'Avg Arousal',
              sortable: true,
              width: 100,
              render: (video) => (
                <Text size="sm" fw={500} c={video.avgArousal > 0.6 ? 'orange' : 'dimmed'}>
                  {video.avgArousal.toFixed(2)}
                </Text>
              )
            },
            {
              accessor: 'totalDuration',
              title: 'Duration',
              sortable: true,
              width: 90,
              render: (video) => (
                <Text size="sm" c="dimmed">
                  {video.totalDuration.toFixed(0)}s
                </Text>
              )
            },
            {
              accessor: 'captionsGenerated',
              title: 'Captions',
              sortable: true,
              width: 100,
              render: (video) => (
                <Group gap={4}>
                  <Badge size="sm" color="green" variant="light">{video.captionsGenerated}</Badge>
                  {video.captionsFailed > 0 && (
                    <Badge size="sm" color="red" variant="light">{video.captionsFailed}</Badge>
                  )}
                </Group>
              )
            },
            {
              accessor: 'actions',
              title: '',
              width: 80,
              render: (video) => (
                <Group gap={4}>
                  <Tooltip label="Play source video">
                    <ActionIcon
                      variant="subtle"
                      color="green"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleVideoPreview(video.name)
                      }}
                    >
                      <IconPlayerPlay size={16} />
                    </ActionIcon>
                  </Tooltip>
                  <Tooltip label="View details">
                    <ActionIcon
                      variant="subtle"
                      color="cerulean"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleRowClick(video)
                      }}
                    >
                      <IconEye size={16} />
                    </ActionIcon>
                  </Tooltip>
                </Group>
              )
            }
          ]}
          sortStatus={sortStatus}
          onSortStatusChange={setSortStatus}
          totalRecords={filteredData.length}
          recordsPerPage={PAGE_SIZE}
          page={page}
          onPageChange={setPage}
          onRowClick={({ record }) => handleRowClick(record)}
          rowStyle={() => ({ cursor: 'pointer' })}
          minHeight={400}
          noRecordsText="No videos found"
        />
      </Paper>

      {/* Detail Modal */}
      <Modal
        opened={detailsOpened}
        onClose={closeDetails}
        title={
          <Group gap="sm">
            <IconVideo size={20} color="#3395f3" />
            <Text fw={600}>{selectedVideo?.name}</Text>
          </Group>
        }
        size="lg"
        centered
      >
        {selectedVideo && (
          <ScrollArea h={450}>
            <Stack gap="lg">
              {/* Stats Grid */}
              <SimpleGrid cols={4}>
                <Paper p="md" bg="dark.7" radius="md" className="stat-card">
                  <Text size="xs" c="dimmed" mb={4}>Total Scenes</Text>
                  <Text size="xl" fw={700} c="cerulean">{selectedVideo.totalScenes}</Text>
                </Paper>
                <Paper p="md" bg="dark.7" radius="md" className="stat-card">
                  <Text size="xs" c="dimmed" mb={4}>With Person</Text>
                  <Text size="xl" fw={700} c="green">{selectedVideo.scenesWithPerson}</Text>
                </Paper>
                <Paper p="md" bg="dark.7" radius="md" className="stat-card">
                  <Text size="xs" c="dimmed" mb={4}>Captions</Text>
                  <Text size="xl" fw={700}>{selectedVideo.captionsGenerated}</Text>
                </Paper>
                <Paper p="md" bg="dark.7" radius="md" className="stat-card">
                  <Text size="xs" c="dimmed" mb={4}>Duration</Text>
                  <Text size="xl" fw={700}>{selectedVideo.totalDuration.toFixed(0)}s</Text>
                </Paper>
              </SimpleGrid>

              {/* Emotion Breakdown */}
              <Box>
                <Text size="sm" fw={600} c="dimmed" mb="md">Emotion Breakdown</Text>
                <Group align="flex-start" gap="xl">
                  {/* Ring Progress */}
                  <Center>
                    <RingProgress
                      size={160}
                      thickness={16}
                      roundCaps
                      sections={Object.entries(selectedVideo.emotionBreakdown)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 5)
                        .map(([emotion, count]) => ({
                          value: (count / selectedVideo.totalScenes) * 100,
                          color: EMOTION_COLORS[emotion] || '#868e96',
                          tooltip: `${emotion}: ${count}`
                        }))}
                      label={
                        <Center>
                          <Stack gap={0} align="center">
                            <Text size="xs" c="dimmed">scenes</Text>
                            <Text fw={700} size="lg">{selectedVideo.totalScenes}</Text>
                          </Stack>
                        </Center>
                      }
                    />
                  </Center>

                  {/* Legend */}
                  <Stack gap="xs" style={{ flex: 1 }}>
                    {Object.entries(selectedVideo.emotionBreakdown)
                      .sort(([,a], [,b]) => b - a)
                      .map(([emotion, count]) => (
                        <Group key={emotion} justify="space-between">
                          <Group gap="xs">
                            <Box
                              w={12}
                              h={12}
                              style={{
                                borderRadius: 2,
                                backgroundColor: EMOTION_COLORS[emotion] || '#868e96'
                              }}
                            />
                            <Text size="sm" tt="capitalize">{emotion}</Text>
                          </Group>
                          <Group gap="xs">
                            <Text size="sm" fw={500}>{count}</Text>
                            <Text size="xs" c="dimmed">
                              ({((count / selectedVideo.totalScenes) * 100).toFixed(1)}%)
                            </Text>
                          </Group>
                        </Group>
                      ))
                    }
                  </Stack>
                </Group>
              </Box>

              {/* Metrics */}
              <SimpleGrid cols={2}>
                <Paper p="md" bg="dark.7" radius="md">
                  <Text size="xs" c="dimmed" mb="xs">Average Valence</Text>
                  <Group justify="space-between" align="center">
                    <Text size="lg" fw={600} c={selectedVideo.avgValence > 0 ? 'green' : 'red'}>
                      {selectedVideo.avgValence.toFixed(3)}
                    </Text>
                    <Progress
                      value={((selectedVideo.avgValence + 1) / 2) * 100}
                      color={selectedVideo.avgValence > 0 ? 'green' : 'red'}
                      size="lg"
                      style={{ width: 100 }}
                    />
                  </Group>
                </Paper>
                <Paper p="md" bg="dark.7" radius="md">
                  <Text size="xs" c="dimmed" mb="xs">Average Arousal</Text>
                  <Group justify="space-between" align="center">
                    <Text size="lg" fw={600}>
                      {selectedVideo.avgArousal.toFixed(3)}
                    </Text>
                    <Progress
                      value={selectedVideo.avgArousal * 100}
                      color="orange"
                      size="lg"
                      style={{ width: 100 }}
                    />
                  </Group>
                </Paper>
              </SimpleGrid>

              {/* Coverage */}
              <Paper p="md" bg="dark.7" radius="md">
                <Text size="xs" c="dimmed" mb="xs">Average Detection Coverage</Text>
                <Progress
                  value={selectedVideo.avgDetectionCoverage * 100}
                  color="cerulean"
                  size="xl"
                  radius="xl"
                />
                <Text size="sm" mt="xs" ta="right">
                  {(selectedVideo.avgDetectionCoverage * 100).toFixed(1)}%
                </Text>
              </Paper>

              {/* Play Video Button */}
              <Box>
                <ActionIcon
                  variant="filled"
                  color="cerulean"
                  size="lg"
                  onClick={() => {
                    closeDetails()
                    handleVideoPreview(selectedVideo.name)
                  }}
                >
                  <IconPlayerPlay size={20} />
                </ActionIcon>
                <Text size="xs" c="dimmed" mt={4}>Play source video</Text>
              </Box>
            </Stack>
          </ScrollArea>
        )}
      </Modal>

      {/* Video Preview Modal */}
      <VideoPreview
        opened={videoPreviewOpened}
        onClose={closeVideoPreview}
        videoUrl={videoPreviewName ? `/api/video/source/${videoPreviewName}` : ''}
        title={videoPreviewName || ''}
        subtitle="Source Video"
      />
    </>
  )
}
